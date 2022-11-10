# encoding: utf-8

import torch, os
from torch.nn.parallel import DataParallel
from utils.util_logger import get_current_logger
from utils.util_config import empty_config_node
from utils.util_config import get_output_dir
from .network_dncnn import DnCNN

def build_model(cfg, is_train=True):
    device = 'cuda' if cfg.TASK.DEVICES is not None else 'cpu'
    model_name = cfg.MODELG.TYPE.lower()
    
    if model_name == 'dncnn':
        model = DnCNN(
            in_channels=cfg.MODELG.IN_CHANNELS,
            mod_channels=cfg.MODELG.MOD_CHANNELS,
            out_channels=cfg.MODELG.OUT_CHANNELS,
            num_layers=cfg.MODELG.NUM_LAYERS,
            act_mode=cfg.MODELG.ACT_MODE
        ).to(device)
    # add model

    else:
        raise Exception("Not found this modle!")

    logger = get_current_logger(cfg)

    # pretrain field
    if is_train and not empty_config_node(cfg.MODELG.PRETRAINED):
        model.load_state_dict(torch.load(cfg.MODELG.PRETRAINED))
        logger.info('Pretrained weight: {}'.format(cfg.MODELG.PRETRAINED))

    elif not is_train:
        if empty_config_node(cfg.TEST.WEIGHT):
            raise Exception("Not found this weight!")
        # test for artefact
        model.load_state_dict(torch.load(cfg.TEST.WEIGHT))
        logger.info('Loading weight: {}'.format(cfg.TEST.WEIGHT))

    logger.info("Model: \n{}".format(model))
    
    return DataParallel(model)