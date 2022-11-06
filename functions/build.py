# encoding: utf-8

import torch
from torch import nn
from utils.util_logger import get_current_logger

def build_function(cfg):
    device = 'cuda' if cfg.TASK.DEVICES is not None else 'cpu'
    loss_name = cfg.LOSS.TYPE.lower()

    if loss_name == "l1":
        loss_fn = nn.L1Loss().to(device)
    elif loss_name == 'l2':
        loss_fn = nn.MSELoss().to(device)

    else:
        raise Exception("Not found this functions!")

    logger = get_current_logger(cfg)
    logger.info('Loss function: {}.'.format(loss_name))

    return loss_fn