# encoding: utf-8

from utils.util_config import empty_config_node

def do_parser(cfg, batch_data):
    device = 'cpu' if empty_config_node(cfg.TASK.DEVICES) else 'gpu'
    parser_type = cfg.DATASETS.PARSER.lower()

    if parser_type == 'pair':
        return __parse_pair(batch_data, device)
    # add parser

    else:
        raise Exception("Not found this parser!")

def __parse_pair(batch_data, device):
    # y(low-quality), x(high-quality), None, [y_path, x_path]
    return (batch_data[0].to(device), 
           batch_data[1].to(device), 
           None, 
           (batch_data[2], batch_data[3]))