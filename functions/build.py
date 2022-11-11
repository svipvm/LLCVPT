# encoding: utf-8

import torch
from torch import nn
from utils.util_logger import get_current_logger

def build_function(cfg):
    device = 'cuda' if cfg.TASK.DEVICES is not None else 'cpu'
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