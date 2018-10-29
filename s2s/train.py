#!/usr/bin/env python
# coding=utf-8

import argparse
import pickle
import os
import itertools

import torch

import s2s.config as config
import s2s.models as models
from s2s.utils.dataloader import build_dataloaders
from s2s.utils.vocab import Vocab
from s2s.utils.utils import update_config

def train(args):

    model_path = os.path.join(args.model_root, args.config, args.suffix)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    config_path = os.path.join(model_path, 'config.pkl')
    if args.restore is not None:
        assert os.path.exists(config_path)
        # restore config
        cf = pickle.load(config_path)
    else:
        cf = getattr(config, args.config)(args.data_root, model_path)

    if args.params is not None:
        # update config
        update_config(config=cf, params=args.params)
    # save config
    pickle.dump(cf, open(config_path, 'wb'))

    dc = cf.dataset
    
    # TODO: BPE
    src_vocab = Vocab(model_path, dc.max_vocab_src, dc.min_freq, prefix='src')
    tgt_vocab = Vocab(model_path, dc.max_vocab_tgt, dc.min_freq, prefix='tgt')
    if args.vocab:
        rebuild = True
        src_file = '{0}.{1}'.format(dc.train_prefix, dc.src)
        tgt_file = '{0}.{1}'.format(dc.train_prefix, dc.tgt)
        corpus = dc.build_corpus(src_file)
        corpus.extend(dc.build_corpus(tgt_file))
    else:
        rebuild = False
        corpus = None
    
    src_vocab.build(rebuild, corpus)
    tgt_vocab.build(rebuild, corpus)
    # cf.src_vocab = src_vocab
    # cf.tgt_vocab = tgt_vocab
    cf.src_vocab_size = len(src_vocab)
    cf.tgt_vocab_size = len(tgt_vocab)

    train, dev, test = build_dataloaders(cf, src_vocab, tgt_vocab)

    # model
    model = getattr(models, cf.model)(cf, src_vocab, tgt_vocab)
    optimizer = getattr(torch.optim , cf.optimizer)(model.parameters(), **cf.optimizer_kwargs)

    # restore checkpoints
    if args.restore is not None:
        checkpoint = torch.load(args.restore)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        step = checkpoint['step']
        best_valid_metric = checkpoint['best_valid_metric']
        best_test_metric = checkpoint['best_test_metric']
        best_step = checkpoint['best_step']
    else:
        step = 0
        best_valid_metric = cf.init_metric
        best_test_metric = cf.init_metric
        best_step = 0

    
    stop_train = False

    for epoch in itertools.count():
        for i, batch in enumerate(train):
            loss = model.train_step(batch)
            step += 1
            print('step', step)
            print('train loss', loss)
            # logits = self.forward(batch)
            # preds = model.greedy_decode(batch)

            # eval on dev set
            # dev_loss_all = 0
            for dev_batch in dev:
                dev_loss, dev_batch_siize = model.get_loss(dev, batch)
                # dev_loss_all += dev_loss
                # print('dev loss', dev_loss)

            # eval on test set
            test_loss = model.get_loss(batch)
            print('test loss', dev_loss)


            # save logs

            # save model

            if step >= cf.max_step:
                # stop training
                stop_train = True
                break
            print()
        if stop_train:
            break

    print('Stop training')





