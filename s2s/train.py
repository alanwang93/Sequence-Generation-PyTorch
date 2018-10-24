#!/usr/bin/env python
# coding=utf-8

import argparse
import pickle
import os

import torch

import s2s.config as config
import s2s.models as models
from s2s.utils.dataloader import build_dataloaders
from s2s.utils.vocab import Vocab
from s2s.utils.utils import update_config

def train(args):

    config_path = os.path.join(args.restore, 'config.pkl')
    if args.restore is not None:
        assert os.path.exists(config_path)
        # restore config
        config = pickle.load(config_path)
    else:
        cf = getattr(config, args.config)(args.data_root)

    if args.params is not None:
        # update config
        update_config(config=cf, args.params)
    # save config
    pickle.dump(config_path)

    dc = cf.dataset
    src_vocab = Vocab(args.model_path, dc.max_vocab_src, dc.min_freq, prefix='src')
    tgt_vocab = Vocab(args.model_path, dc.max_vocab_tgt, dc.min_freq, prefix='tgt')

    if not os.path.exists(os.path.join(args.model_path, 'src_vocab.pkl')) or args.vocab:
        # rebuild vocab
        rebuild = True
    else:
        rebuild = False

    src_vocab.build(rebuild=rebuild)
    tgt_vocab.build(rebuild=rebuild)
    # cf.src_vocab = src_vocab
    # cf.tgt_vocab = tgt_vocab
    cf.src_vocab_size = len(src_vocab)
    cf.tgt_vocab_size = len(tgt_vocab)

    train, dev, test = build_dataloaders(cf, src_vocab, tgt_vocab)

    # model
    model = getattr(models, cf.model)(cf, src_vocab, tgt_vocab)
    optimizer = getattr(torch.optim , cf.optimizer)(cf.optimizer_kwargs)

    # restore checkpoints
    if args.restore is not None:
        cp_path = os.path.join(args.restore, 'model.pkl')
        assert os.path.exists(cp_path)
        checkpoint = torch.load(cp_path)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        best_valid_metric = checkpoint['best_valid_metric']
        best_test_metric = checkpoint['best_test_metric']
        best_step = checkpoint['best_step']
    else:
        epoch = 1
        step = 1
        best_valid_metric = cf.init_metric
        best_test_metric = cf.init_metric
        best_step = 1



    for _ in range(200):
        for i, batch in enumerate(train):
            loss = model.train_step(batch)
            print(i, loss)
            # logits = self.forward(batch)
            preds = model.greedy_decode(batch)