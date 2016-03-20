"""
__file__

    param_config.py

__description__

    This file provides global parameter configurations for the project.

__author__

    based on script of Chenglong Chen < c.chenglong@gmail.com >
    modified by Andrea Schioppa

"""

import os
import numpy as np


############
## Config ##
############
class ParamConfig:
    def __init__(self, code_dir, data_dir, output_dir,
                 feat_dir, fig_dir, subm_dir):
        self.code_dir = code_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.feat_dir = feat_dir
        self.fig_dir = fig_dir
        self.subm_dir = subm_dir


my_code_dir = '/Users/schioand/leave_academia/kaggle/bnp-paribas/code'
config = ParamConfig(my_code_dir, data_dir = ''.join([my_code_dir, '/../data/']),
                     output_dir = ''.join([my_code_dir, '/../output/']),
                     feat_dir = ''.join([my_code_dir, '/../feat/']),
                     fig_dir = ''.join([my_code_dir, '/../fig/']),
                     subm_dir = ''.join([my_code_dir, '/../subm/']))
