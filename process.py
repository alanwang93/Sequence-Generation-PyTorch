#!/usr/bin/env python
# coding=utf-8

import argparse
import os

from multiprocessing import Process
import data_config
from utils.vocab import Vocab

def main(args):
    assert args.config is not None
    dc = getattr(data_config, args.config)()

    if args.root is not None:
        dc.root = args.root

    if args.vocab:
        vocab = Vocab(data_config=dc)
        vocab._init_vocab()
        for src_file, tgt_file in zip(dc.train_src, dc.train_tgt):
            print("Build vocab from {0}, {1}".format(src_file, tgt_file))
            corpus = dc.build_corpus(src_file, tgt_file)
            # Build Vocabualry
            vocab.update(corpus)
        vocab._finish()
    else:
        print("Load vocab")
        vocab = Vocab(data_config=dc)
        vocab.build(rebuild=False)

    vocab.summary()

    if args.extract:
        for (src, tgt) in [(dc.train_src, dc.train_tgt), \
                (dc.valid_src, dc.valid_tgt), (dc.test_src, dc.test_tgt)]:
            print("Build {0}, {1}".format(src, tgt))
            dc.build_data(src, tgt, vocab)
        
        #print("Extracting features")
        #file_list = dc.train_list + dc.valid_list + dc.test_list
        #processes = [Process(target=dc.extract, args=(f, vocab)) for f in file_list]
        #for p in processes:
        #    p.start()
        #for p in processes:
        #    p.join()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='data config name')
    parser.add_argument('--vocab', dest='vocab', action='store_true', \
            help='(Re)build vocabulary')
    # parser.add_argument('--embed', dest='embed', action='store_true')
    parser.add_argument('--extract', dest='extract', action='store_true')
    parser.add_argument('--root', default=None, help='root directory')
    args = parser.parse_args()
    main(args)




