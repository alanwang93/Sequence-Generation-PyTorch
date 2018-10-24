#!/usr/bin/env python
# coding=utf-8

import os

def update_config(config, params):
    """
    params: string of form `param1 value1 param2 value2 ...`
    """
    params = params.split(' ')
    assert len(params) % 2 == 0

    for i in range(len(params)//2):
        param = params[2*i]
        value = params[2*i+1]

        if hasattr(config, param):
            setattr(config, param, value)
        else:
            raise AttributeError()

def init_logging(log_name):
    """
    """
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S'   )
    handler = logging.FileHandler(log_name)
    out = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    out.setFormatter(formatter)
    out.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logging.getLogger().addHandler(out)
    logging.getLogger().setLevel(logging.INFO)
    return logging

def save_checkpoint(obj, is_best):
    
