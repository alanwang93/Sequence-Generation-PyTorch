#!/usr/bin/env python
# coding=utf-8

import argparse
import pickle
import os

import s2s.config as config
import s2s.models as models
from s2s.utils.dataloader import build_dataloaders
from s2s.utils.vocab import Vocab
from s2s.utils.utils import update_config

def train(args):

    config_path = os.path.join(args.restore, 'config.pkl')
    if args.restore is not None and os.path.exists(config_path):
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

    for _ in range(200):
        for i, batch in enumerate(train):
            loss = model.train_step(batch)
            print(i, loss)
            # logits = self.forward(batch)
            preds = model.greedy_decode(batch)