#!/usr/bin/env python
# coding=utf-8

import os
import logging
import sys
import math

from rouge import Rouge
import torch.nn as nn
import torch.nn.init as init

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
            raise ValueError()

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


def init_weights_by_type(m):
    """
    Specially initialization for RNNs, etc.
    """
    print('Initialze', m, )
    if type(m) == nn.GRU:
        for i in range(m.num_layers):
            K = 3
            func = nn.init.orthogonal_
            for k in range(K):
                s = k * m.hidden_size
                e = (k+1) * m.hidden_size
                w = getattr(m, 'weight_ih_l{0}'.format(i))[s:e]
                func(w)
                w = getattr(m, 'weight_hh_l{0}'.format(i))[s:e]
                func(w)
                if m.bidirectional:
                    w = getattr(m, 'weight_ih_l{0}_reverse'.format(i))[s:e]
                    func(w)
                    w = getattr(m, 'weight_hh_l{0}_reverse'.format(i))[s:e]
                    func(w)

    elif type(m) == nn.GRUCell:
        pass
    elif type(m) == nn.Linear:
        pass
    else:
        raise ValueError('Module type not recgnized: {0}'.format(type(m)))


def init_weights(param, name=None):
    dim = param.dim()
    if dim == 1:
        param.normal_(0, math.sqrt(6 / (1 + param.size(0))))
    elif dim == 2:
        param = init.xavier_normal_(param, gain=1.)
    else:
        raise ValueError('Wrong dimension: {0}'.format(dim))
    print('Initialize {0}'.format(None))
    




