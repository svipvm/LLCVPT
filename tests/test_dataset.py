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


class TestDataSet(unittest.TestCase):

    def test_dataset(self):
        cfg.merge_from_file("configs/config_dncnn.yaml")
        print('open this config file:\n{}'.format(cfg))

        generate_time_stamp(cfg)
        logger = setup_logger(cfg)

        train_loader = build_data_loader(cfg, 0)
        
        d_iter = iter(train_loader)
        data_pair = next(d_iter)
        logger.info("size: (batch, channels, patch, patch)")
        logger.info("L:" + str(data_pair[0].shape))
        logger.info("H:" + str(data_pair[1].shape))
        logger.info("LP:" + str(data_pair[2].shape))
        logger.info("HP:" + str(data_pair[3].shape))

        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
