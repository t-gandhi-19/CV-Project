import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


_C = CfgNode()
cfg = _C

'''
----------------------------- Dataset options -------------------------------
Dataset options
    DATASET: Victim dataset
    DATA_DIR: Victim dataset root directory
    DO_LOWER_CASE: Whether to lower case
    CACHE_DIR: Cache directory
'''

_C.DATASET = CfgNode()
_C.DATASET.DATASET = 'imagenet32'
_C.DATASET.DATA_DIR = 'data/tp'
_C.DATASET.DO_LOWER_CASE = True
_C.DATASET.CACHE_DIR = 'cache'
_C.DATASET.RECTIFIED = False

'''
----------------------------- TASK options -------------------------------
TASK options
    DEPTH_EXTIMATE: Whether to estimate depth
    COLOR_MODELS: Color models to use
    3D_VIEW : Whether to use 3D view
    WINDOW_SIZE: Window size
    SEARCH_SIZE: Search size
    SAVE_DISPARITY_MAP: Whether to save disparity
'''


_C.TASK = CfgNode()
_C.TASK.DEPTH_ESTIMATE = False
_C.TASK.COLOR_MODELS = ['RGB']
_C.TASK.VIEW_3D = False
_C.TASK.WINDOW_SIZE = 15
_C.TASK.SEARCH_SIZE = 100
_C.TASK.SAVE_DISPARITY_MAP = False
_C.TASK.TWO_IMAGES_STEREO = False
_C.TASK.MULTIPLE_IMAGES_STEREO = True


# --------------------------------- Default config -------------------------- #
_C.OUT_DIR = "results"


_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def merge_from_file(cfg_file):
    """Merges config from a file."""
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)

def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)

def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)

def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    cfg.SAVE_DIR = f'{_C.OUT_DIR}'
    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()
