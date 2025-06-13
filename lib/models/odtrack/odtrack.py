import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
from lib.models.odtrack.vit import vit_base_patch16_224, vit_large_patch16_224
# from lib.models.odtrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.odtrack.vit_ce_FreqFusion import vit_large_patch16_224_ce, vit_base_patch16_224_ce  # xyl
from lib.utils.box_ops import box_xyxy_to_cxcywh

from functools import partial

# from dynamixer.models.vision_model import VisionModel, VisionBlock
# from dynamixer.models.dynamixer import DynaMixerBlock
# ============================================================================= xyl
from .hgit.HGINet import HGIT, feature2token, token2feature
from .mefc.DRSformer_arch import subnet


class ODTrack(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=1):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        # track query: save the history information of the previous frame
        self.track_query_v = None
        self.track_query_i = None
        self.token_len = token_len
        
        # Fuse RGB and T search regions, random initialized
        hidden_dim = transformer.embed_dim
        self.rgbt_fuse_search = conv(hidden_dim * 2, hidden_dim)  
        # self.hgit=HGIT(nn.BatchNorm2d, dim=hidden_dim, num_clusters=8, dropout=0.1)
        self.mefc = subnet(dim=hidden_dim * 2, steps=4)  # 专家数量为：steps * num_ops = 4 * 8

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        assert isinstance(search, list), "The type of search is not List"

        out_dict = []
        for i in range(len(search)):
            x, aux_dict = self.backbone(z=template.copy(), x=search[i],
                                        ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, track_query_v=self.track_query_v, track_query_i=self.track_query_i, token_len=self.token_len)
            feat_last = x
            if isinstance(x, list):
                feat_last = torch.cat(x, dim=0)
                # feat_last = x[-1]
                
            # enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            if self.backbone.add_cls_token:
                self.track_query_v = (x[:, :self.token_len].clone()).detach() # stop grad  (B, N, C)
                self.track_query_i = (x[:, self.token_len + ce_template_mask.shape[1] + self.feat_len_s:self.token_len + self.token_len + ce_template_mask.shape[1] + self.feat_len_s].clone()).detach()
            # att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            # opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            
            # Forward head
            frame_z = len(template)
            out = self.forward_head(cat_feature=feat_last, frame=frame_z, gt_score_map=None)

            out.update(aux_dict)
            out['backbonwe_feat'] = x # test = 1+z+x
            
            out_dict.append(out)
            
        return out_dict

    # def forward_head(self, opt, gt_score_map=None):
    def forward_head(self, cat_feature, frame, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        if cat_feature.shape[1] < 1027: # 256版本的cat_feature是torch.Size([1, 1026, 768])
            num_template_token = 64 # 256
            num_search_token = 256 # 256
            # print("------当前版本为256版本，lib/models/odtrack/odtrack.py 102行--------")
        else:                           # 384版本的cat_feature是torch.Size([1, 2306, 768])
            num_template_token = 144 # 384
            num_search_token = 576 # 384
            # print("------当前版本为384版本，lib/models/odtrack/odtrack.py 102行--------")

        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        enc_opt1 = cat_feature[:, self.token_len + num_template_token * frame:self.token_len + num_template_token * frame + num_search_token, :]  # torch.Size([1, 256, 768])
        enc_opt2 = cat_feature[:, -num_search_token:, :]  # torch.Size([1, 256, 768])
        # enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)  # torch.Size([1, 256, 1536])  # 原始的
        # =================================================================================================xyl HGIT的特征融合
        if 0:
            enc_opt1_feat = token2feature(enc_opt1)  # torch.Size([1, 768, 16, 16])
            enc_opt2_feat = token2feature(enc_opt2)  # torch.Size([1, 768, 16, 16])
            enc_opt1_feat_hgit, enc_opt2_feat_hgit = self.hgit(enc_opt1_feat, enc_opt2_feat)  # torch.Size([1, 768, 16, 16])
            enc_opt1_token_hgit = feature2token(enc_opt1_feat_hgit)  # torch.Size([1, 256, 768])
            enc_opt2_token_hgit = feature2token(enc_opt2_feat_hgit)  # torch.Size([1, 256, 768])
            enc_opt = torch.cat([enc_opt1_token_hgit, enc_opt2_token_hgit], dim=2)  # torch.Size([1, 256, 1536])
        else:
            enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)  # torch.Size([1, 256, 1536])  # 原始的
        # =================================================================================================xyl HGIT的特征融合

        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 1, 2)).contiguous()  # torch.Size([1, 1, 256, 1536])
        bs, Nq, HW, C = opt.size()
        HW = int(HW/2)
        opt_feat = opt.view(-1, self.feat_sz_s, self.feat_sz_s, C)  # torch.Size([1, 16, 16, 1536])      
        # opt_feat = self.rgbt_fuse_search(opt_feat.permute((0, 3, 1, 2)))  #  torch.Size([1, 1536, 16, 16]) -> torch.Size([1, 768, 16, 16])  xyl
        # =================================================================================================xyl MEFC的模态加权
        if 1:
            opt_feat_mefc = self.mefc(opt_feat.permute((0, 3, 1, 2)))  # [1, 16, 16, 1536] > [1, 1536, 16, 16] > torch.Size([1, 1536, 16, 16])
            opt_feat = self.rgbt_fuse_search(opt_feat_mefc)
        else:
            opt_feat = self.rgbt_fuse_search(opt_feat.permute((0, 3, 1, 2)))
        # =================================================================================================xyl MEFC的模态加权
        
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True) # pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map) # score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError



def build_odtrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('ODTrack' not in cfg.MODEL.PRETRAIN_FILE) and training: # ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,)

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, 
                                         add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                         attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, 
                                         )
        
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            )

    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = ODTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
    )

    if 'ODTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_file, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        if 0:
            # 如果checkpoint是一个字典，可能包含了模型权重、优化器状态等
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['net'])
                print("从checkpoint中加载模型权重")
            else:
                # 直接加载模型权重
                model.load_state_dict(checkpoint['net'])
                print("直接加载模型权重")

            # 只保存模型参数
            torch.save(model.state_dict(), '/home/xyl/newdrive/xyl-code2/0.my_trackers/FMTrack/output/ODTrack_ep0014.pth.tar')
            print(f"模型权重已保存到 {'/home/xyl/newdrive/xyl-code2/0.my_trackers/FMTrack/output'}")
    return model
