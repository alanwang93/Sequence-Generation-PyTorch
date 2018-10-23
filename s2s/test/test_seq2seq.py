#!/usr/bin/env python
# coding=utf-8

import argparse

import s2s.config as config
from s2s.utils.dataloader import build_dataloaders
from s2s.utils.vocab import Vocab
from s2s.models import Seq2seq

def test(args):
    cf = getattr(config, 'Test')(args.data_root)
    dc = cf.dataset
    src_vocab = Vocab(dc.path, dc.max_vocab_src, dc.min_freq, prefix='src')
    tgt_vocab = Vocab(dc.path, dc.max_vocab_tgt, dc.min_freq, prefix='tgt')
    src_vocab.build(rebuild=False)
    tgt_vocab.build(rebuild=False)
    cf.src_vocab = src_vocab
    cf.tgt_vocab = tgt_vocab
    cf.src_vocab_size = len(src_vocab)
    cf.tgt_vocab_size = len(tgt_vocab)

    train, dev, test = build_dataloaders(cf, src_vocab, tgt_vocab)
    # iter_train = iter(train)
    

    # model
    model = Seq2seq(cf, src_vocab, tgt_vocab)
    
    print('Test training')
    for _ in range(200):
        for i, batch in enumerate(train):
            loss = model.train_step(batch)
            print(i, loss)
            # logits = self.forward(batch)
            preds = model.greedy_decode(batch)
            # print(i, preds[:4])

    # print('Test greedy decoding')
    # iter_train = iter(train)
    # for i, batch in enumerate(iter_train):
    #     preds = model.greedy_decode(batch)
    #     print(i, preds)
        



def main(args):
    test(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=None)
    args = parser.parse_args()
    main(args)
