#!/usr/bin/env python
# coding=utf-8

import os
import logging
import sys

from rouge import Rouge

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
            setattr(config, param, type(getattr(config, param))(value))
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

def to(obj, device):
    if isinstance(obj, dict):
        for k in obj.keys():
            obj[k] = obj[k].to(device)
    else:
        obj = obj.to(device)
    return obj


def rouge_score(hyps, refs, mode='f'):
    metrics = dict()
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    metrics['ROUGE-1'] = scores['rouge-1'][mode]
    metrics['ROUGE-2'] = scores['rouge-2'][mode]
    metrics['ROUGE-L'] = scores['rouge-l'][mode]
    return metrics


def compute_metrics(hyps, refs, names, avg=True):
    metrics = dict()
    if 'rouge' in names:
        metrics.update(rouge_score(hyps, refs))
    return metrics
