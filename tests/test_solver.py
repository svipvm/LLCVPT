# encoding: utf-8


import sys, unittest
sys.path.append('.')

import torch
from torchsummary import summary

from config import cfg
from utils.util_logger import *
from utils.util_file import *
from utils.util_config import *
from data import build_data_loader
from models import build_model
from solvers import build_optimizer
from solvers import build_scheduler
from functions import build_function


class TestSolver(unittest.TestCase):

    def test_solver(self):
        cfg.merge_from_file("configs/config_dncnn.yaml")
        print('open this config file:\n{}'.format(cfg))

        generate_time_stamp(cfg)
        logger = setup_logger(cfg)
        
        model = build_model(cfg)
        optimzier = build_optimizer(cfg, model)
        logger.info("" + str(optimzier))

        scheduler = build_scheduler(cfg, optimzier)
        logger.info("" + str(scheduler))

        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
