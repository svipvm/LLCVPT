# encoding: utf-8


import sys, unittest
sys.path.append('.')

from config import cfg
from utils.util_logger import *
from utils.util_file import *
from utils.util_config import *


class TestConfig(unittest.TestCase):

    def test_config(self):
        cfg.merge_from_file("configs/config_dncnn.yaml")
        print('open this config file:\n{}'.format(cfg))

        generate_time_stamp(cfg)
        cfg.freeze()
        print("output:", get_output_dir(cfg))
        record_config_file(cfg)

        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
