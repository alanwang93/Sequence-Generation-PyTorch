#!/usr/bin/env python
# coding=utf-8

import argparse
from s2s.train import train
from s2s.train_biclf import train as train_biclf

def main(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'train_biclf':
        train_biclf(args)
    elif args.mode == 'test':
        pass
    else:
        raise ValueError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--suffix', type=str, default='default')
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--params', type=str, default=None)
    
    parser.add_argument('--cuda', type=int, default=None)
    
    parser.add_argument('--rebuild', dest='rebuild', action='store_true', \
            help='Rebuild vocab and dataset')
    parser.add_argument('--model_root', type=str, default='checkpoints')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--raw_root', type=str, default='raw')

    args = parser.parse_args()
    main(args)
