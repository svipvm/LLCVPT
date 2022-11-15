# encoding: utf-8

import argparse, sys
sys.path.append('.')

from config import cfg
from utils.util_config import *
from utils.util_file import *
from utils.util_option import *
from utils.util_sys import *
from utils.util_logger import setup_logger

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

    load_device_info(cfg)
    
    fixed_random_seed(cfg)

    from models import build_model
    model = build_model(cfg)

    from data import build_data_loader
    train_loader = build_data_loader(cfg, 0)
    valid_loader = build_data_loader(cfg, 1)

    from solvers import build_optimizer
    optimizer = build_optimizer(cfg, model)

    from solvers import build_scheduler
    scheduler = build_scheduler(cfg, optimizer)

    from functions import build_function
    loss_fn = build_function(cfg)

    logger.info("The description of this task is: {}".format(cfg.TASK.VERSION))
    from workers.plain_trainer import do_train
    do_train(cfg, model, train_loader, valid_loader, optimizer, scheduler, loss_fn)
    logger.info("This result was saved to: {}".format(get_output_dir(cfg)))

if __name__ == "__main__":
    main()
