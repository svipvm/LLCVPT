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
from solver import build_optimizer
from functions import build_function


class TestMethod(unittest.TestCase):

    def test_method(self):
        cfg.merge_from_file("configs/config_dncnn.yaml")
        print('open this config file:\n{}'.format(cfg))

        generate_time_stamp(cfg)
        logger = setup_logger(cfg)
        
        device = torch.device('cuda')
        model = build_model(cfg).to(device)
        # logger.info('' + str(model))

        # summary(model, input_size=(1, 64, 64))

        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
