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
