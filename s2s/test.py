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
from s2s.utils.utils import update_config, init_logging, to, compute_metrics
from s2s.utils.summary import Summary

def test(args):
    assert args.restore is not None

    model_path = os.path.join(args.model_root, args.config + '_' + args.dataset, args.suffix)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger = init_logging(os.path.join(model_path, 'test_log.txt'))

    config_path = os.path.join(model_path, 'config.pkl')
    # restore config
    cf = pickle.load(open(config_path, 'rb'))

    #if args.params is not None:
    #    # update config
    #    update_config(config=cf, params=args.params)
    # save config
    #pickle.dump(cf, open(config_path, 'wb'))
    logger.info('Config name: {0}\n{1}'.format(args.config, pprint.pformat(vars(cf), 2)))
    dc = cf.dataset
    logger.info('Data config: \n{0}'.format(pprint.pformat(vars(dc), 2)))

    if args.cuda is None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.cuda)
    
    # TODO: BPE or WordPiece
    src_vocab = Vocab(dc.path, dc.max_vocab_src, dc.min_freq, prefix='src')
    tgt_vocab = Vocab(dc.path, dc.max_vocab_tgt, dc.min_freq, prefix='tgt')

    corpus = None
    
    src_vocab.build(args.vocab, corpus)
    tgt_vocab.build(args.vocab, corpus)
    cf.src_vocab_size = len(src_vocab)
    cf.tgt_vocab_size = len(tgt_vocab)

    train, dev, test = build_dataloaders(cf, src_vocab, tgt_vocab, False, False)
    n_dev, n_test = map(len, [dev, test])
    logger.info('Dataset sizes: dev {0}, test {1}'.format(n_dev, n_test))

    # model
    model = getattr(models, cf.model)(cf, src_vocab, tgt_vocab, args.restore)
    model = model.to(device)

    # restore checkpoints
    checkpoint = torch.load(args.restore, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])
    step = checkpoint['step']
    

    dev_loss = 0
    n_dev_batch = 0
    model.eval()
    dev_hyps, dev_refs = [], []
    for dev_batch in dev:
        dev_batch = to(dev_batch, device)
        dev_loss_, n_dev_batch_ = model.get_loss(dev_batch)
        dev_loss += dev_loss_*n_dev_batch_
        n_dev_batch += n_dev_batch_
        preds = model.greedy_decode(dev_batch)
        for ref, hyp in zip(dev_batch['tgt_in'], preds):
            ref = ' '.join(tgt_vocab.tos(ref))
            hyp = ' '.join(tgt_vocab.tos(hyp))
            if hyp == '':
                hyp = ' '
            dev_refs.append(ref)
            dev_hyps.append(hyp)
    dev_loss /= n_dev_batch
    
    ## dev summary
    dev_metrics = {'loss':dev_loss}
    names = ['rouge']
    dev_metrics.update(compute_metrics(dev_hyps, dev_refs, names))

    logger.info('[Dev] Step {0}, Loss: {1:.5f}'.format(step, dev_loss))
    for k, v in dev_metrics.items():
        logger.info('[Dev]\t{0}: {1:.5f}'.format(k, v))
    logger.info('[Dev] Sample:\nref: {0}\nhyp: {1}\n'.format(dev_refs[0], dev_hyps[0]))


    # eval on test set
    test_loss = 0
    n_test_batch = 0
    model.eval()
    test_hyps, test_refs = [], []
    for test_batch in test:
        test_batch = to(test_batch, device)
        test_loss_, n_test_batch_ = model.get_loss(test_batch)
        test_loss += test_loss_*n_test_batch_
        n_test_batch += n_test_batch_
        preds = model.greedy_decode(test_batch)
        for ref, hyp in zip(test_batch['tgt_in'], preds):
            ref = ' '.join(tgt_vocab.tos(ref))
            hyp = ' '.join(tgt_vocab.tos(hyp))
            if hyp == '':
                hyp = ' '
            test_refs.append(ref)
            test_hyps.append(hyp)
    test_loss /= n_test_batch
    
    # test summary
    test_metrics = {'loss':test_loss}
    names = ['rouge']
    test_metrics.update(compute_metrics(test_hyps, test_refs, names))

    logger.info('[Test] Step {0}, Loss: {1:.5f}'\
            .format(step, test_loss))
    for k, v in test_metrics.items():
        logger.info('[Test]\t{0}: {1:.5f}'.format(k, v))

    fout = open('checkpoints/test.out', 'w')
    for i in range(len(test_refs)):
        #logger.info('[Test] Sample:\nref: {0}\nhyp: {1}\n'.format(test_refs[i], test_hyps[i]))
        fout.write('{0}\t{1}\n'.format(test_refs[i], test_hyps[i]))



