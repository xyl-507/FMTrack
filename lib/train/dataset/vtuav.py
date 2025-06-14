import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class VTUAV(BaseVideoDataset):
    """ VTUAV dataset for RGB-T tracking.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the vtuav training data.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'test'.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().vtuav_dir if root is None else root
        super().__init__('VTUAV_add', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list(split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'vtuav'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1][:-1],
                                       'motion_class': meta_info[6].split(': ')[-1][:-1],
                                       'major_class': meta_info[7].split(': ')[-1][:-1],
                                       'root_class': meta_info[8].split(': ')[-1][:-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1][:-1]})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self, split):
        if split == 'test':
            with open(os.path.join(self.root, '..', 'testingsetList.txt')) as f:
                dir_list = list(csv.reader(f))
        else:
            with open(os.path.join(self.root, '..', 'trainingsetList.txt')) as f:
                dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        # bb_anno_file = os.path.join(seq_path, "init.txt")
        bb_anno_file = os.path.join(seq_path, "rgb.txt")  # 需要修改，VTUAV没有init.txt，以rgb.txt为真实值
        gt = pandas.read_csv(bb_anno_file, delimiter=' ', header=None, dtype=np.float32, na_filter=False, low_memory=False).values  # 需要修改,分割符是空格
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover>0).byte()

        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = torch.ByteTensor([1 for v in range(len(bbox))])  # 根据由 bbox 得到的 visible 来选择训练图片的序号，见 ODTrack-RGBT/lib/train/data/sampler.py 114行

        visible_ratio = torch.ByteTensor([1 for v in range(len(bbox))])

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_path, frame_id):
        vis_frame_names = sorted(os.listdir(os.path.join(seq_path, 'rgb')))  # 修改文件夹名字
        inf_frame_names = sorted(os.listdir(os.path.join(seq_path, 'ir')))  # 修改文件夹名字

        # return os.path.join(seq_path, 'visible', vis_frame_names[frame_id]), os.path.join(seq_path, 'infrared', inf_frame_names[frame_id])
        return os.path.join(seq_path, 'rgb', vis_frame_names[frame_id*10]), os.path.join(seq_path, 'ir', inf_frame_names[frame_id*10])  # 修改文件夹名字；frame_id*10是指隔10帧标注，需要加载图片的序号也要隔10帧

    def _get_frame(self, seq_path, frame_id):
        path = self._get_frame_path(seq_path, frame_id)
        return np.concatenate((self.image_loader(path[0]),self.image_loader(path[1])), 2)

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta
