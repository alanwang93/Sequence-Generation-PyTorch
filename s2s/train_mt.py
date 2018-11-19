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
import s2s.utils.dataloader as dataloader
from s2s.utils.vocab import Vocab
from s2s.utils.utils import update_config, init_logging, to, compute_metrics, init_weights
from s2s.utils.summary import Summary

def train(args):
    """
    Multi-task training
    """

    model_path = os.path.join(args.model_root, args.config+'_'+args.dataset, args.suffix)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger = init_logging(os.path.join(model_path, 'train_log.txt'))
    logger.info('Start train.py\n' + '=' * 80)
    logger.info('Args:\n{0}'.format(pprint.pformat(vars(args), 2)))


    config_path = os.path.join(model_path, 'config.pkl')
    if args.restore is not None:
        assert os.path.exists(config_path)
        # restore config
        cf = pickle.load(open(config_path, 'rb'))
    else:
        cf = getattr(config, args.config)(
                args.raw_root, args.data_root, model_path, args.dataset)

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
    
    # TODO: BPE or WordPiece
    src_vocab = Vocab(dc.path, dc.max_vocab_src, dc.min_freq, prefix='src')
    tgt_vocab = Vocab(dc.path, dc.max_vocab_tgt, dc.min_freq, prefix='tgt')

    if args.vocab:
        train_prefix = os.path.join(dc.raw, dc.train_prefix)
        src_file = '{0}.{1}'.format(train_prefix, dc.src)
        tgt_file = '{0}.{1}'.format(train_prefix, dc.tgt)
        corpus = dc.build_corpus(src_file)
        corpus.extend(dc.build_corpus(tgt_file))
    else:
        corpus = None
    
    src_vocab.build(args.vocab, corpus)
    tgt_vocab.build(args.vocab, corpus)
    cf.src_vocab_size = len(src_vocab)
    cf.tgt_vocab_size = len(tgt_vocab)

    dataloader_builder = getattr(dataloader, dc.dataloader)
    train, dev, test = dataloader_builder(cf, src_vocab, tgt_vocab, args.rebuild)
    n_train, n_dev, n_test = map(len, [train, dev, test])
    logger.info('Dataset sizes: train {0}, dev {1}, test {2}'.format(n_train, n_dev, n_test))
    
    # load raw data for evaluation
    dev_refs = []
    with open(os.path.join(dc.raw, dc.dev_prefix + '.' + dc.tgt)) as f:
        for line in f:
            line = line.lower().strip() if dc.lower else line.strip()
            dev_refs.append(line)
    test_refs = []
    with open(os.path.join(dc.raw, dc.test_prefix + '.' + dc.tgt)) as f:
        for line in f:
            line = line.lower().strip() if dc.lower else line.strip()
            test_refs.append(line)


    # model
    model = getattr(models, cf.model)(cf, src_vocab, tgt_vocab, args.restore)
    model = model.to(device)

    logger.info('Parameters')
    for name, param in model.named_parameters():
        logger.info('{0}, size: {1}'.format(name, param.size()))


    # restore checkpoints
    if args.restore is not None:
        checkpoint = torch.load(args.restore)
        model.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state'])
        step = checkpoint['step']
        best_dev_metric = checkpoint['best_dev_metric']
        best_test_metric = checkpoint['best_test_metric']
        best_step = checkpoint['best_step']
        restart = False
    else:
        for name, param in model.named_parameters():
            init_weights(param, name)
        step = 0
        best_dev_metric = cf.init_metric
        best_test_metric = cf.init_metric
        best_step = 0
        restart = True
    
    # load summary
    train_summ = Summary(model_path, prefix='train', restart=restart)
    dev_summ = Summary(model_path, prefix='dev', restart=restart)
    test_summ = Summary(model_path, prefix='test', restart=restart)
    best_summ = Summary(model_path, prefix='best', restart=restart)
    
    stop_train = False
    train_loss = 0
    n_train_batch = 0
    best_metrics = dict()
    train_mt_loss = 0
    
    for _ in itertools.count():
        for i, train_batch in enumerate(train):

            # train
            epoch = step//n_train
            new_mt_coef = cf.mt_coef * (0.5)**epoch
            if model.mt_coef != new_mt_coef:
                model.mt_coef = new_mt_coef
                print('MT coef: ', model.mt_coef)
            model.train()
            train_batch = to(train_batch, device)
            train_loss_, train_mt_loss_, n_train_batch_ = model.train_step(train_batch)
            train_loss += train_loss_*n_train_batch_
            train_mt_loss += train_mt_loss_*n_train_batch_
            n_train_batch += n_train_batch_
            step += 1

            if step % cf.log_freq == 0:

                train_loss /= n_train_batch
                train_mt_loss /= n_train_batch
                # train summary
                train_metrics = {
                        'loss':train_loss,
                        'mt_loss': train_mt_loss }
                train_summ.write(step, train_metrics)
                logger.info('[Train] Step {0}, Loss: {1:.5f}, MT loss: {2:.5f}'.\
                        format(step, train_loss, train_mt_loss))

                train_loss = 0
                train_mt_loss = 0
                n_train_batch = 0


            # eval on dev set
            if step % cf.eval_freq == 0:
                dev_loss = 0
                dev_mt_loss = 0
                n_dev_batch = 0
                model.eval()
                dev_hyps  = []
                for dev_batch in dev:
                    dev_batch = to(dev_batch, device)
                    dev_loss_, dev_mt_loss_, n_dev_batch_ = model.get_loss(dev_batch)
                    dev_loss += dev_loss_*n_dev_batch_
                    dev_mt_loss += dev_mt_loss_*n_dev_batch_
                    n_dev_batch += n_dev_batch_
                    preds = model.greedy_decode(dev_batch)
                    for hyp in preds:
                        #ref = ' '.join(tgt_vocab.tos(ref))
                        hyp = ' '.join(tgt_vocab.tos(hyp))
                        if hyp == '':
                            hyp = ' '
                        #dev_refs.append(ref)
                        dev_hyps.append(hyp)

                dev_loss /= n_dev_batch
                dev_mt_loss /= n_dev_batch
                
                # dev summary
                dev_metrics = {
                        'loss':dev_loss,
                        'mt_loss': dev_mt_loss}
                names = ['rouge']
                dev_metrics.update(compute_metrics(dev_hyps, dev_refs, names))
                dev_summ.write(step, dev_metrics)
                

                logger.info('[Dev] Step {0}, Loss: {1:.5f}, MT loss: {2:.5f}'.\
                        format(step, dev_loss, dev_mt_loss))
                for k, v in dev_metrics.items():
                    logger.info('[Dev]\t{0}: {1:.5f}'.format(k, v))

                logger.info('[Dev] Sample:\nref: {0}\nhyp: {1}\n'.format(dev_refs[0], dev_hyps[0]))


                # eval on test set
                test_loss = 0
                test_mt_loss = 0
                n_test_batch = 0
                model.eval()
                test_hyps = []
                for test_batch in test:
                    test_batch = to(test_batch, device)
                    test_loss_, test_mt_loss_, n_test_batch_ = model.get_loss(test_batch)
                    test_loss += test_loss_*n_test_batch_
                    test_mt_loss += test_mt_loss_*n_test_batch_
                    n_test_batch += n_test_batch_
                    preds = model.greedy_decode(test_batch)
                    for hyp in preds:
                        hyp = ' '.join(tgt_vocab.tos(hyp))
                        if hyp == '':
                            hyp = ' '
                        test_hyps.append(hyp)
                test_loss /= n_test_batch
                test_mt_loss /= n_test_batch
                
                # test summary
                test_metrics = {
                        'loss':test_loss,
                        'mt_loss': test_mt_loss }
                names = ['rouge']
                test_metrics.update(compute_metrics(test_hyps, test_refs, names))
                test_summ.write(step, test_metrics)

                logger.info('[Test] Step {0}, Loss: {1:.5f}, MT loss: {2:.5f}'\
                        .format(step, test_loss, test_mt_loss))
                for k, v in test_metrics.items():
                    logger.info('[Test] {0}: {1:.5f}'.format(k, v))


                model.scheduler.step(dev_metrics[cf.metric])
                logger.info(model.optimizer)
                if cf.is_better(dev_metrics[cf.metric], best_dev_metric):
                    best_step = step
                    best_dev_metric = dev_metrics[cf.metric]
                    best_test_metric = test_metrics[cf.metric]
                    for name in dev_metrics:
                        best_metrics['dev_' + name] = dev_metrics[name]
                    for name in dev_metrics:
                        best_metrics['test_' + name] = test_metrics[name]

                    best_summ.write(best_step, best_metrics)
                    logger.info('[Best] Step {0}, new best'.format(step))
                    for k, v in best_metrics.items():
                        logger.info('[Best] {0}: {1:.5f}'.format(k, v))


                    checkpoint = dict(
                            state_dict=model.state_dict(),
                            optimizer_state=model.optimizer.state_dict(),
                            step=step,
                            best_dev_metric=best_dev_metric,
                            best_test_metric=best_test_metric,
                            best_step=best_step)
                    
                    cp_path = os.path.join(model_path, 'best.pkl')
                    torch.save(checkpoint, cp_path)
                    logger.info('[Save] Model saved to {0}'.format(cp_path))


                best_metrics['dev_' + cf.metric] = best_dev_metric
                best_metrics['test_' + cf.metric] = best_test_metric

                # best up to now
                logger.info('[Best] Up to now, Best step: {0}'.format(best_step))
                for k in best_metrics:
                    logger.info('[Best] {0}: {1:5f}'.format(k, best_metrics[k]))

                dev_loss = 0
                n_dev_batch = 0

                checkpoint = dict(
                        state_dict=model.state_dict(),
                        optimizer_state=model.optimizer.state_dict(),
                        step=step,
                        best_dev_metric=best_dev_metric,
                        best_test_metric=best_test_metric,
                        best_step=best_step)
                
                cp_path = os.path.join(model_path, 'model.pkl')
                torch.save(checkpoint, cp_path)
                logger.info('[Save] Model saved to {0}'.format(cp_path))

            if step % n_train == 0:
                # save model for this epoch
                checkpoint = dict(
                        state_dict=model.state_dict(),
                        optimizer_state=model.optimizer.state_dict(),
                        step=step,
                        best_dev_metric=best_dev_metric,
                        best_test_metric=best_test_metric,
                        best_step=best_step)
                cp_path = os.path.join(model_path, 'epoch_{0}.pkl'.format(epoch))
                torch.save(checkpoint, cp_path)
                logger.info('[Save] Model saved to {0}'.format(cp_path))

            if step >= cf.max_step:
                # stop training
                stop_train = True
                break
        if stop_train:
            break

    print('Stop training')





