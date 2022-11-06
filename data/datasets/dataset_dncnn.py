# encoding: utf-8

import sys
sys.path.append(".")

import random, torch
import numpy as np
import torch.utils.data as data
from utils import util_img as util
from . import DatasetType


class DatasetDnCNN(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # -----------------------------------------
    """

    def __init__(self, cfg, data_type):
        super(DatasetDnCNN, self).__init__()
        self.data_type = data_type
        self.img_channels = cfg.TASK.IMG_CHANNELS

        if self.data_type == DatasetType.DATASETS_TRAIN.value:
            self.patch_size = cfg.DATASETS.TRAIN.PATCH_SIZE
            self.sigma = cfg.DATASETS.TRAIN.SIGMA
            self.paths_H = util.get_image_paths(cfg.DATASETS.TRAIN.H_DATASETS)
        elif self.data_type == DatasetType.DATASETS_VALID.value:
            self.sigma = cfg.DATASETS.VALID.SIGMA
            self.paths_H = util.get_image_paths(cfg.DATASETS.VALID.H_DATASETS)
        elif self.data_type == DatasetType.DATASETS_TEST.value:
            self.sigma = cfg.DATASETS.TEST.SIGMA
            self.paths_H = util.get_image_paths(cfg.DATASETS.TEST.H_DATASETS)


    def __getitem__(self, index):
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.img_channels)

        L_path = H_path

        if self.data_type == DatasetType.DATASETS_TRAIN.value:
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()

            # --------------------------------
            # add noise
            # --------------------------------
            noise = torch.randn(img_L.size()).mul_(self.sigma/255.0)
            img_L.add_(noise)

        else:
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)

            # --------------------------------
            # add noise
            # --------------------------------
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma/255.0, img_L.shape)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)

    def __str__(self):
        return "DnCNN {}: Denosing on AWGN with fixed sigma.".format(DatasetType(self.data_type))
