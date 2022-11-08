# encoding: utf-8

import argparse, sys
sys.path.append('.')

from config import cfg
from utils.util_config import *
from utils.util_file import *
from utils.util_option import *
from utils.util_logger import setup_logger
from models import build_model
from data import build_data_loader
from functions import build_function
from solvers import build_optimizer
from solvers import build_scheduler
from trainers import plain_trainer


def main():
    parser = argparse.ArgumentParser(description="DnCNN Train Project")
    parser.add_argument("--config", 
                default="configs/config_dncnn.yaml", 
                help="path to config file", 
                type=str)
    parser.add_argument("opts", 
                default=None, 
                help="modify config options", 
                nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config != "":
        cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    generate_time_stamp(cfg)
    logger = setup_logger(cfg)
    record_config_file(cfg)
    logger.info("Running with config:\n{}".format(cfg))

    fixed_random_seed(cfg)

    model = build_model(cfg)
    train_loader = build_data_loader(cfg, 0)
    valid_loader = build_data_loader(cfg, 1)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn = build_function(cfg)

    logger.info("The description of this task is: {}".format(cfg.TASK.VERSION))
    plain_trainer(cfg, model, train_loader, valid_loader, optimizer, scheduler, loss_fn)
    logger.info("This result was saved to: {}".format(get_output_dir(cfg)))

if __name__ == "__main__":
    main()
