#!/usr/bin/env python
# coding=utf-8

import argparse
import pickle
import os
import itertools
import pprint

import torch
import numpy as np

import s2s.config as config
import s2s.models as models
from s2s.utils.dataloader import build_dataloaders
from s2s.utils.vocab import Vocab
from s2s.utils.utils import update_config, init_logging, to
from s2s.utils.summary import Summary

def train(args):

    model_path = os.path.join(args.model_root, args.config, args.suffix)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger = init_logging(os.path.join(model_path, 'train_log.txt'))
    logger.info('Start train.py\n' + '=' * 80)
    logger.info('Args:\n{0}'.format(pprint.pformat(vars(args), 2)))


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
    logger.info('Config name: {0}\n{1}'.format(args.config, pprint.pformat(vars(cf), 2)))
    dc = cf.dataset
    logger.info('Data config: \n{0}'.format(pprint.pformat(vars(dc), 2)))

    if args.cuda is None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.cuda)
    
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
    n_train, n_dev, n_test = map(len, [train, dev, test])

    # model
    model = getattr(models, cf.model)(cf, src_vocab, tgt_vocab)
    model = model.to(device)
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
        restart = False
    else:
        step = 0
        best_valid_metric = cf.init_metric
        best_test_metric = cf.init_metric
        best_step = 0
        restart = True
    
    # load summary
    train_summ = Summary(model_path, prefix='train', restart=restart)
    dev_summ = Summary(model_path, prefix='dev', restart=restart)
    test_summ = Summary(model_path, prefix='test', restart=restart)
    
    stop_train = False
    train_loss = 0
    n_train_batch = 0

    for _ in itertools.count():
        for i, train_batch in enumerate(train):

            # train
            train_batch = to(train_batch, device) 
            train_loss_, n_train_batch_ = model.train_step(train_batch)
            train_loss += train_loss_*n_train_batch_
            n_train_batch += n_train_batch_
            step += 1
            epoch = step//n_train

            if step % cf.log_freq == 0:
                # train summary
                train_metrics = {'loss':train_loss}
                train_summ.write(step, train_metrics)

                train_loss /= n_train_batch
                logger.info('[Train] Step {0}, Loss: {1:.5f}'.format(step, train_loss))

                train_loss = 0
                n_train_batch = 0

            # logits = self.forward(batch)
            # preds = model.greedy_decode(batch)

            # eval on dev set
            if step % cf.eval_freq == 0:
                dev_loss = 0
                n_dev_batch = 0

                for dev_batch in dev:
                    dev_batch = to(dev_batch, device)
                    dev_loss_, n_dev_batch_ = model.get_loss(dev_batch)
                    dev_loss += dev_loss_*n_dev_batch_
                    n_dev_batch += n_dev_batch_
                dev_loss /= n_dev_batch
                
                # dev summary
                dev_metrics = {'loss':dev_loss}
                dev_summ.write(step, dev_metrics)

                logger.info('[Dev] Step {0}, Loss: {1:.5f}'.format(step, dev_loss))

                dev_loss = 0
                n_dev_batch = 0

                checkpoint = dict(
                        state_dict=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        step=step,
                        best_valid_metric=best_valid_metric,
                        best_test_metric=best_test_metric,
                        best_step=best_step)
                
                cp_path = os.path.join(model_path, 'model.pkl')
                torch.save(checkpoint, cp_path)
                logger.info('[Save] Model saved to {0}'.format(cp_path))

            if step % n_train == 0:
                # save model for this epoch
                cp_path = os.path.join(model_path, 'epoch_{0}.pkl'.format(epoch))
                torch.save(checkpoint, cp_path)
                logger.info('[Save] Model saved to {0}'.format(cp_path))
               


                # dev_loss_all += dev_loss
                # print('dev loss', dev_loss)

            # eval on test set
            #test_loss = model.get_loss(batch)
            #print('test loss', dev_loss)


            # save logs

            # save model

            if step >= cf.max_step:
                # stop training
                stop_train = True
                break
        if stop_train:
            break

    print('Stop training')





