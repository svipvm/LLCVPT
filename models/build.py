# encoding: utf-8

import torch, os, functools
from torch.nn.parallel import DataParallel
from utils.util_logger import get_current_logger
from utils.util_config import empty_config_node
from .network_dncnn import DnCNN
from torch.nn import init

def build_model(cfg, is_train=True):
    device = 'cpu' if empty_config_node(cfg.TASK.DEVICES) else 'cuda'
    model_type = cfg.MODELG.TYPE.lower()
    
    if model_type == 'dncnn':
        model = DnCNN(
            in_channels=cfg.MODELG.IN_CHANNELS,
            mod_channels=cfg.MODELG.MOD_CHANNELS,
            out_channels=cfg.MODELG.OUT_CHANNELS,
            num_layers=cfg.MODELG.NUM_LAYERS,
            act_mode=cfg.MODELG.ACT_MODE,
            bias=cfg.MODELG.BIAS
        )
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
    # init model weight
    elif is_train:
        init_type = cfg.MODELG.INIT_TYPE
        init_bn_type = cfg.MODELG.INIT_BN_TYPE
        gain = cfg.MODELG.INIT_GAIN
        init_fn = functools.partial(__init_weight, init_type, init_bn_type, gain)
        model.apply(init_fn)
        logger.info('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(
            init_type, init_bn_type, gain))

    logger.info("Model: \n{}".format(model))

    return DataParallel(model).to(device)

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

def __init_weight(model,  init_type, init_bn_type, gain):
    classname = model.__class__.__name__

    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if init_type == 'normal':
                init.normal_(model.weight.data, 0, 0.1)
                model.weight.data.clamp_(-1, 1).mul_(gain)
        # add

        else:
            raise Exception("Not found this init type")

    elif classname.find('BatchNorm2d') != -1:
        if init_bn_type == 'uniform':  # preferred
                if model.affine:
                    init.uniform_(model.weight.data, 0.1, 1.0)
                    init.constant_(model.bias.data, 0.0)
        # add

        else:
            raise Exception("Not found this init bn type")
    