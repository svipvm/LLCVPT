# encoding: utf-8

import argparse, os, sys, time, shutil, torch
sys.path.append('.')

from config import cfg
from utils.util_logger import *
from utils.util_config import *
from models import build_model
from data import build_data_loader
from workers.plain_tester import do_test


def main():
    parser = argparse.ArgumentParser(description="DnCNN Train Project")
    parser.add_argument("--config_file", 
                default="configs/config_dncnn.yaml", 
                help="path to config file", 
                type=str)
    parser.add_argument("opts", 
                default=None, 
                help="modify config options", 
                nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    generate_time_stamp(cfg)
    logger = setup_logger(cfg)
    record_config_file(cfg)
    logger.info("Running with config:\n{}".format(cfg))

    model = build_model(cfg, False)
    test_loader = build_data_loader(cfg, 2)

    logger.info("The description of this task is: {}".format(cfg.TASK.VERSION))
    do_test(cfg, model, test_loader)
    logger.info("This result was saved to: {}".format(get_output_dir(cfg)))

if __name__ == "__main__":
    main()
