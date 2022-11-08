# encoding: utf-8


import sys, unittest
sys.path.append('.')

from config import cfg
from utils.util_logger import *
from utils.util_file import *
from utils.util_config import *
from data import build_data_loader
from models import build_model
from solvers import build_optimizer
from functions import build_function


class TestMethod(unittest.TestCase):

    def test_method(self):
        cfg.merge_from_file("configs/config_dncnn.yaml")
        print('open this config file:\n{}'.format(cfg))

        generate_time_stamp(cfg)
        # logger = setup_logger(cfg, 0)
        
        train_loader = build_data_loader(cfg, 0)
        print(train_loader)
        test_loader = build_data_loader(cfg, 1)
        print(test_loader)

        # model = build_model(cfg)
        # print(model)

        # optimizer = build_optimizer(cfg, model)
        # print(optimizer)

        # loss_fn = build_function(cfg)
        # print(loss_fn)

        # for idx, item in enumerate(train_loader):
        #     print(idx, item)
        #     break

        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
