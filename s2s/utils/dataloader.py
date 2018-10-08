#!/usr/bin/env python
# coding=utf-8

import torch
import numpy as np
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

#class BatchIterator:
#    def __init__(self, generator, data_size):
#       self.iterator = generator()
 
class DictDataset(data.Dataset):
    """
    Build dataset with list of dicts
    """
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


def dict_collate_fn(batch):
    keys = ['src', 'tgt', 'len_src', 'len_tgt']
    data = {}
    for k in keys:
        data[k] = []
    for d in batch:
        for k in keys:
            data[k].append(torch.tensor(d[k]))
    data['src'] = pad_sequence(data['src'], True, 0)
    data['tgt'] = pad_sequence(data['tgt'], True, 0)
    data['len_src'] = torch.stack(data['len_src'])
    data['len_tgt'] = torch.stack(data['len_tgt'])

    return data



def build_dataloaders(config, src_vocab, tgt_vocab):
    cf = config
    dc = config.dataset # data config
    dataloaders = []
    for prefix in [dc.train_prefix,
                   dc.dev_prefix,
                   dc.test_prefix]:
        
        src_file = prefix + '.' + dc.src
        tgt_file = prefix + '.' + dc.tgt
        examples = []
        
        with open(src_file, 'r') as s, open(tgt_file, 'r') as t:
            for src, tgt in zip(s, t):
                str_src = src.strip().split(' ')
                str_tgt = tgt.strip().split(' ')
                len_src = len(str_src)
                len_tgt = len(str_tgt)
                src = src_vocab.toi(str_src)
                tgt = tgt_vocab.toi(str_tgt)

                example = {
                        'src': src, 
                        'tgt': tgt,
                        'len_src': len_src,
                        'len_tgt': len_tgt,
                        #'str_src': str_src,
                        #'str_tgt': str_tgt
                        }
                examples.append(example)

            if prefix in [dc.train_prefix, dc.dev_prefix]:
                dataloader = data.DataLoader(
                    DictDataset(examples),
                    batch_size=cf.batch_size,
                    shuffle=True,
                    num_workers=4,
                    collate_fn=dict_collate_fn)
            else:
                dataloader = data.DataLoader(
                    DictDataset(examples),
                    batch_size=cf.batch_size,
                    shuffle=False,
                    num_workers=4,
                    collate_fn=dict_collate_fn)

            dataloaders.append(dataloader)
    
    train, dev, test = dataloaders
    
    return train, dev, test







