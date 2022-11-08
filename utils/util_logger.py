# encoding: utf-8

import logging, os, sys
from .util_config import get_output_dir

class Logger:
    def __init__(self, cfg):
        self.logger_name = cfg.TASK.NAME
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        save_file_name = os.path.join(get_output_dir(cfg), "{}.log".format(self.logger_name))
        fh = logging.FileHandler(save_file_name, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger

def setup_logger(cfg, distributed_rank = 0):
    # if distributed_rank > 0:
    #     return logger
    logger = Logger(cfg).get_logger()
    return logger

def get_current_logger(cfg):
    logger_name = cfg.TASK.NAME
    return logging.getLogger(logger_name)