# encoding: utf-8

import torch
from torch import nn
from utils.util_logger import get_current_logger
from utils.util_config import empty_config_node

def build_function(cfg):
    device = 'cpu' if empty_config_node(cfg.TASK.DEVICES) else 'gpu'
    loss_type = cfg.LOSS.TYPE.lower()

    if loss_type == "l1":
        loss_fn = nn.L1Loss().to(device)
    elif loss_type == 'l2':
        loss_fn = nn.MSELoss().to(device)
    # add loss function

    else:
        raise Exception("Not found this functions!")

    logger = get_current_logger(cfg)
    logger.info('Loss function: {}.'.format(loss_type))

    return loss_fn