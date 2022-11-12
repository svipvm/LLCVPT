# encoding: utf-8

import torch, os
from torch.nn.parallel import DataParallel
from utils.util_logger import get_current_logger
from utils.util_config import empty_config_node
from utils.util_config import get_output_dir
from .network_dncnn import DnCNN

def build_model(cfg, is_train=True):
    device = 'cuda' if cfg.TASK.DEVICES is not None else 'cpu'
    model_type = cfg.MODELG.TYPE.lower()
    
    if model_type == 'dncnn':
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
        load_model(cfg.MODELG.PRETRAINED, model)
        logger.info('Pretrained weight: {}'.format(cfg.MODELG.PRETRAINED))
    # test for artefact
    elif not is_train:
        if empty_config_node(cfg.TEST.WEIGHT):
            raise Exception("Not found this weight!")
        load_model(cfg.TEST.WEIGHT, model)
        logger.info('Loading weight: {}'.format(cfg.TEST.WEIGHT))

    logger.info("Model: \n{}".format(model))
    
    return DataParallel(model)

def save_model(model, path):
    if isinstance(model, DataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, path)

def load_model(path, model, strict=True):
    if isinstance(model, DataParallel):
        model = model.module
    state_dict = torch.load(path)
    model.load_state_dict(state_dict, strict=strict)
    