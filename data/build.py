# encoding: utf-8

from torch.utils import data

from utils.util_logger import get_current_logger
from .datasets.dataset_dncnn import DatasetDnCNN
from .transforms import build_transform
from .collate_batch import *
from .datasets import DatasetType


def build_data_loader(cfg, data_type):
    if data_type == DatasetType.DATASETS_TRAIN.value:
        batch_size = cfg.DATASETS.TRAIN.BATCH_SIZE
        shuffle = True
        num_workers = cfg.DATASETS.TRAIN.NUM_WORKERS
        drop_last = True
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1
        drop_last = False
 
    transforms = build_transform(cfg, data_type)
    dataset = __get_dataset(cfg, transforms, data_type)
    collate_fn = __get_collate(cfg)

    data_loader = data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True)

    return data_loader

def __get_dataset(cfg, transforms, data_type):
    dataset_type = cfg.DATASETS.TYPE.lower()

    if dataset_type == "dncnn":
        dataset =  DatasetDnCNN(cfg, data_type)
    # add dataset

    else:
        raise Exception("Not found this dataset!")

    logger = get_current_logger(cfg)
    logger.info("{}".format(dataset))

    return dataset

def __get_collate(cfg):
    dataset_type = cfg.DATASETS.TYPE.lower()

    plain_dataset_list = ['dncnn']
    
    if dataset_type in plain_dataset_list:
        return plain_collate_fn

    else:
        return None
        # raise Exception("Not found this collate function!")