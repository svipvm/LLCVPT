# encoding: utf-8

from yacs.config import CfgNode
from .util_file import *

def get_output_dir(cfg):
    if cfg.RECORD.TIME_STAMP is None:
        raise Exception("To generate timestap, please!")
    return mkdir_if_not_exist([
        cfg.RECORD.OUTPUT_DIR, 
        cfg.TASK.NAME,
        cfg.RECORD.TIME_STAMP
    ])

def generate_time_stamp(cfg):
    if cfg.RECORD.TIME_STAMP : return
    try:
        cfg.defrost()
        cfg.RECORD.TIME_STAMP = datetime.datetime.now().strftime("%YY_%mM_%dD_%HH_%MM_%SS_%f")
        cfg.freeze()
    except:
        raise Exception("Failure to generate timestap!")

def empty_config_node(cnode):
    return not cnode

def record_config_file(cfg):
    with open(os.path.join(get_output_dir(cfg), 'config.yaml'), 'w') as f:
        def enumerate_node(cnode, pre_space):
            for k_, v_ in cnode.items():
                if isinstance(v_, CfgNode):
                    f.write("{}{}:\n".format(pre_space, k_))
                    enumerate_node(v_, pre_space + (' ' * 2))
                else:
                    if empty_config_node(v_): continue
                    f.write("{}{}: {}\n".format(pre_space, k_, v_))
                    # f.write(pre_space + str(k_) + ": " + str(v_) + "\n")
        enumerate_node(cfg, '')
