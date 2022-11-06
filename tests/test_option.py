# encoding: utf-8

import sys, unittest
sys.path.append('.')

import random, torch, numpy

from config import cfg
from utils.util_logger import *
from utils.util_file import *
from utils.util_config import *
from utils.util_option import *


class TestOption(unittest.TestCase):

    def test_option(self):
        cfg.merge_from_file("configs/config_dncnn.yaml")
        print('open this config file:\n{}'.format(cfg))

        generate_time_stamp(cfg)
        cfg.freeze()
        print("output:", get_output_dir(cfg))
        
        fixed_random_seed(cfg)

        print("random {}, numpy {}, torch {}".format(
            random.random(),
            numpy.random.random(),
            torch.randn((1, 1))))

        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
