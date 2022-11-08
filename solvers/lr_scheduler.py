# encoding: utf-8

from torch.optim import lr_scheduler
from utils.util_logger import get_current_logger

def build(cfg, optimizer):
    scheduler_name = cfg.SCHEDULER.TYPE.lower()
    scheduler_list = ['multisteplr']
    if scheduler_name not in scheduler_list: return
        # raise Exception("Not found [{}] Scheduler!".format(scheduler_name))
    
    scheduler = None
    if scheduler_name == 'multisteplr':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, cfg.SCHEDULER.MILESTONES, cfg.SCHEDULER.GAMMA)

    logger = get_current_logger(cfg)
    logger.info("Scheduler: {}.".format(scheduler_name))

    return scheduler