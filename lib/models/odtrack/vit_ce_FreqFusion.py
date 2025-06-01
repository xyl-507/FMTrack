import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock
from .hgit.HGINet import feature2token, token2feature
from .FreqFusion.FreqFusion import FreqFusion

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None, add_cls_token=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.add_cls_token = add_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(  # lib/models/layers/attn_blocks.py 中 注释了CE模块
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.FreqFusion = FreqFusion(hr_channels=embed_dim, lr_channels=embed_dim,
                   scale_factor=1, lowpass_kernel=5, highpass_kernel=3, up_group=1,
                   upsample_mode='nearest', align_corners=False,
                   feature_resample=False,
                   feature_resample_group=8,
                   hr_residual=True,
                   comp_feat_upsample=False,  # True
                   compressed_channels=(embed_dim + embed_dim) // 8,
                   use_high_pass=True, use_low_pass=True, semi_conv=False, feature_resample_norm=True)

        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, track_query_v=None, track_query_i=None,
                         token_type="add", token_len=1
                         ):

        
        B, H, W = x[0].shape[0], x[0].shape[2], x[0].shape[3]

        H_x_re, W_x_re = H/16, W/16
        
        x_v = self.patch_embed(x[0])
        x_i = self.patch_embed(x[1])
        

        z_v = []
        z_i = []
        # 循环遍历 z 中的每个元素，并执行 stack dim=1 操作
        for tensor in z:
        # 对当前元素执行 stack dim=1 操作，并将结果添加到 z 中
            z_v.append(torch.stack([tensor[0]], dim=1))
            z_i.append(torch.stack([tensor[1]], dim=1))
        # 将所有结果连接起来，形成一个新的张量
        z_v = torch.cat(z_v, dim=1)
        z_i = torch.cat(z_i, dim=1)
        
        # z_v = torch.stack(z[0], dim=1)
        # z_i = torch.stack(z[1], dim=1)
        _, T_z, C_z, H_z, W_z = z_v.shape
        H_z_re, W_z_re = H_z/16, W_z/16
        
        z_v = z_v.flatten(0, 1)
        z_v = self.patch_embed(z_v)
        z_i = z_i.flatten(0, 1)
        z_i = self.patch_embed(z_i)
        
        
        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            if token_type == "concat":
                if track_query_v is None:
                    query_v = self.cls_token.expand(B, token_len, -1)
                    query_i = self.cls_token.expand(B, token_len, -1)
                else:
                    track_len = track_query_v.size(1)
                    new_query_v = self.cls_token.expand(B, token_len - track_len, -1)
                    new_query_i = self.cls_token.expand(B, token_len - track_len, -1)
                    query_v = torch.cat([new_query_v, track_query_v], dim=1)
                    query_i = torch.cat([new_query_i, track_query_i], dim=1)
            elif token_type == "add":
                new_query_v = self.cls_token.expand(B, token_len, -1)  # copy B times
                new_query_i = self.cls_token.expand(B, token_len, -1)  # copy B times
                query_v = new_query_v if track_query_v is None else track_query_v + new_query_v
                query_i = new_query_i if track_query_i is None else track_query_i + new_query_i
            query_v = query_v + self.cls_pos_embed
            query_i = query_i + self.cls_pos_embed
        
        # z = z + self.pos_embed_z
        # x = x + self.pos_embed_x
        
        # Visible and infrared data share the positional encoding and other parameters in ViT
        z_v += self.pos_embed_z
        x_v += self.pos_embed_x
        z_i += self.pos_embed_z
        x_i += self.pos_embed_x

        if self.add_sep_seg:
            x = x + self.search_segment_pos_embed
            z = z + self.template_segment_pos_embed

        if T_z > 1:  # multiple memory frames
            # z = z.view(B, T_z, -1, z.size()[-1]).contiguous()
            # z = z.flatten(1, 2)
            z_v = z_v.view(B, T_z, -1, z_v.size()[-1]).contiguous()  # torch.Size([48, 64, 768]) -> torch.Size([16, 3, 64, 768])
            z_v = z_v.flatten(1, 2)  # torch.Size([16, 192, 768])
            
            z_i = z_i.view(B, T_z, -1, z_i.size()[-1]).contiguous()  # torch.Size([48, 64, 768]) -> torch.Size([16, 3, 64, 768])
            z_i = z_i.flatten(1, 2)  # torch.Size([16, 192, 768])

        lens_z = z_v.shape[1] # lens_z = z.shape[1]  # HW  192
        lens_x = x_v.shape[1] # lens_x = x.shape[1]  # HW  256

        # x = combine_tokens(z, x, mode=self.cat_mode)  # (B, z+x, 768)
        x_v = combine_tokens(z_v, x_v, mode=self.cat_mode)  # torch.Size([16, 448, 768])  OSTrack: torch.Size([1, 320, 768])
        x_i = combine_tokens(z_i, x_i, mode=self.cat_mode)  # torch.Size([16, 448, 768])  OSTrack: torch.Size([1, 320, 768])
        
        # if self.add_cls_token:
        #     x = torch.cat([query, x], dim=1)     # (B, 1+z+x, 768)
        #     query_len = query.size(1)
        # x = self.pos_drop(x)
        if self.add_cls_token:
            x_v = torch.cat([query_v, x_v], dim=1)     # (B, 1+z+x, 768)  # torch.Size([16, 449, 768]) OSTrack: torch.Size([1, 321, 768])
            x_i = torch.cat([query_i, x_i], dim=1)     # (B, 1+z+x, 768)  # torch.Size([16, 449, 768]) OSTrack: torch.Size([1, 321, 768])
            query_len = query_v.size(1)
        x_v = self.pos_drop(x_v)
        x_i = self.pos_drop(x_i)
        
        
        # global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        # global_index_t = global_index_t.repeat(B, 1)
        # global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        # global_index_s = global_index_s.repeat(B, 1)
        
        global_index_t_v = torch.linspace(0, lens_z - 1, lens_z).to(x_v.device)
        global_index_t_v = global_index_t_v.repeat(B, 1)
        global_index_s_v = torch.linspace(0, lens_x - 1, lens_x).to(x_v.device)
        global_index_s_v = global_index_s_v.repeat(B, 1)
        
        global_index_t_i = torch.linspace(0, lens_z - 1, lens_z).to(x_i.device)
        global_index_t_i = global_index_t_i.repeat(B, 1)
        global_index_s_i = torch.linspace(0, lens_x - 1, lens_x).to(x_i.device)
        global_index_s_i = global_index_s_i.repeat(B, 1)
        
        
        removed_indexes_s_v = []
        removed_indexes_s_i = []
        mamba_index = 0
        
        for i, blk in enumerate(self.blocks):
            if self.add_cls_token:  # cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = True   # x_v: torch.Size([1, 321, 768])
                x_v, global_index_t_v, global_index_s_v, removed_index_s_v, attn_v = \
                    blk(x_v, global_index_t_v, global_index_s_v, mask_x, ce_template_mask, ce_keep_rate, 
                        add_cls_token=self.add_cls_token, query_len=query_len)
                x_i, global_index_t_i, global_index_s_i, removed_index_s_i, attn_i = \
                    blk(x_i, global_index_t_i, global_index_s_i, mask_x, ce_template_mask, ce_keep_rate, 
                        add_cls_token=self.add_cls_token, query_len=query_len)
                # --------------------------------------------------------------------------------------------- xyl FreqFusion的特征融合
                # tbsi_loc = [3, 6, 9]
                tbsi_loc = [6, 9]
                # tbsi_loc = range(12)
                if tbsi_loc is not None and i in tbsi_loc:
                    # x_v_ori = x_v
                    # recover x_v to go through hgit blocks
                    lens_z_v_new = lens_z
                    lens_x_v_new = lens_x
                    z_v_tokens = x_v[:, :lens_z_v_new]  # torch.Size([1, 192, 768])
                    x_v_tokens = x_v[:, lens_z_v_new:lens_z_v_new+lens_x_v_new]  # torch.Size([1, 256, 768])，最后一个是cls_token
                    x_v_feat = token2feature(x_v_tokens)  # torch.Size([1, 768, 16, 16])

                    # x_i_ori = x_i
                    # recover x_i to go through hgit blocks
                    z_i_tokens = x_i[:, :lens_z_v_new]  # torch.Size([1, 192, 768])
                    x_i_tokens = x_i[:, lens_z_v_new:lens_z_v_new+lens_x_v_new]  # torch.Size([1, 256, 768])
                    x_i_feat = token2feature(x_i_tokens)  # torch.Size([1, 768, 16, 16])

                    _, x_v_hgit, x_i_hgit = self.FreqFusion(x_v_feat, x_i_feat)  # torch.Size([1, 768, 16, 16])

                    x_v_hgit_token = feature2token(x_v_hgit)  # torch.Size([1, 256, 768])
                    x_v = combine_tokens(z_v_tokens, x_v_hgit_token, mode=self.cat_mode)  # torch.Size([1, 320, 768])

                    x_i_hgit_token = feature2token(x_i_hgit)  # torch.Size([1, 256, 768])
                    x_i = combine_tokens(z_i_tokens, x_i_hgit_token, mode=self.cat_mode)  # torch.Size([1, 448, 768])

                    x_v = torch.cat([query_v, x_v], dim=1)     # (B, 1+z+x, 768)  # torch.Size([1, 449, 768])
                    x_i = torch.cat([query_i, x_i], dim=1)     # (B, 1+z+x, 768)  # torch.Size([1, 449, 768])
                # --------------------------------------------------------------------------------------------- xyl FreqFusion的特征融合
            else:
                x_v, global_index_t_v, global_index_s_v, removed_index_s_v, attn_v = \
                    blk(x_v, global_index_t_v, global_index_s_v, mask_x, ce_template_mask, ce_keep_rate, add_cls_token=self.add_cls_token)
                    
                x_i, global_index_t_i, global_index_s_i, removed_index_s_i, attn_i = \
                    blk(x_i, global_index_t_i, global_index_s_i, mask_x, ce_template_mask, ce_keep_rate, add_cls_token=self.add_cls_token)
                
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s_v.append(removed_index_s_v)
                removed_indexes_s_i.append(removed_index_s_i)

        x_v = self.norm(x_v)
        x_i = self.norm(x_i)
        lens_x_new = global_index_s_v.shape[1]
        lens_z_new = global_index_t_v.shape[1]

        if self.add_cls_token:
            query_v = x_v[:, :query_len]
            z_v = x_v[:, query_len:lens_z_new+query_len]
            x_v = x_v[:, lens_z_new+query_len:]
            
            query_i = x_i[:, :query_len]
            z_i = x_i[:, query_len:lens_z_new+query_len]
            x_i = x_i[:, lens_z_new+query_len:]
            
        else:
            z_v = x_v[:, :lens_z_new]
            x_v = x_v[:, lens_z_new:]
            
            z_i = x_i[:, :lens_z_new]
            x_i = x_i[:, lens_z_new:]

        if removed_indexes_s_v and removed_indexes_s_v[0] is not None:
            removed_indexes_cat_v = torch.cat(removed_indexes_s_v, dim=1)
            pruned_lens_x = lens_x - lens_x_new
            pad_x_v = torch.zeros([B, pruned_lens_x, x_v.shape[2]], device=x_v.device)
            x_v = torch.cat([x_v, pad_x_v], dim=1)
            index_all_v = torch.cat([global_index_s_v, removed_indexes_cat_v], dim=1)
            # recover original token order
            C = x_v.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x_v = torch.zeros_like(x_v).scatter_(dim=1, index=index_all_v.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x_v)
            
            
            removed_indexes_cat_i = torch.cat(removed_indexes_s_i, dim=1)
            pruned_lens_x = lens_x - lens_x_new
            pad_x_i = torch.zeros([B, pruned_lens_x, x_i.shape[2]], device=x_i.device)
            x_i = torch.cat([x_i, pad_x_i], dim=1)
            index_all_i = torch.cat([global_index_s_i, removed_indexes_cat_i], dim=1)
            # recover original token order
            C = x_i.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x_i = torch.zeros_like(x_i).scatter_(dim=1, index=index_all_i.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x_i)
            
            

        x_v = recover_tokens(x_v, lens_z_new, lens_x, mode=self.cat_mode)
        x_i = recover_tokens(x_i, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x_v = torch.cat([query_v, z_v, x_v], dim=1)
        x_i = torch.cat([query_i, z_i, x_i], dim=1)

        x = torch.cat([x_v, x_i], dim=1)
        
        # aux_dict = {}
        aux_dict = {
            "attn_v": attn_v,
            "attn_i": attn_i,
            "removed_indexes_s_v": removed_indexes_s_v,  # used for visualization
            "removed_indexes_s_i": removed_indexes_s_i,
        }

        return self.norm(x), aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, return_last_attn=False, track_query_v=None, track_query_i=None,
                token_type="add", token_len=1):
        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                            track_query_v=track_query_v, track_query_i=track_query_i, token_type=token_type, token_len=token_len)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            try:
                checkpoint = torch.load(pretrained, map_location="cpu")
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
                print("missing keys:", missing_keys)
                print("unexpected keys:", unexpected_keys)
                print('Load pretrained model from: ' + pretrained)
            except:
                print("Warning: MAE Pretrained model weights are not loaded !")

    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
