# encoding: utf-8

import torch.optim as optim

from utils.util_logger import get_current_logger


def build_optimizer(cfg, model):
    solver_name = cfg.SOLVER.TYPE.lower()
    solver_list = ['sgd', 'adam']
    if solver_name not in solver_list:
        raise Exception("Not found [{}] optimizer!".format(solver_name))
        
    params = []
    for key, param in model.named_parameters():
        if not param.requires_grad: continue
        params.append(param)
    
    if solver_name == 'sgd':
        optimizer = None
    elif solver_name == 'adam':
        optimizer = optim.Adam(params=params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    
    logger = get_current_logger(cfg)
    logger.info("Optimizer: {}.".format(solver_name))

    return optimizer

