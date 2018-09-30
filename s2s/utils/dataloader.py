#!/usr/bin/env python
# coding=utf-8

import torch
import numpy as np

#class BatchIterator:
#    def __init__(self, generator, data_size):
#       self.iterator = generator()
 
class DictDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


def build_dataloaders(config, vocab):
    cf = config
    dc = config.dataset # data config
    dataloaders = []
    for prefix in [self.train.prefix,
                   self.dev_prefix,
                   self.test_prefix]:
        
        src_file = prefix + '.' + dc.src
        tgt_file = prefix + '.' + dc.tgt
        examples = []
        
        with open(src_file, 'r') as s, open(tgt_file, 'r') as t:
            str_src = src.strip().split(' ')
            str_tgt = tgt.strip().split(' ')
            len_src = len(str_src)
            len_tgt = len(str_tgt)
            src = vocab.toi(str_src)
            tgt = vocab.toi(str_tgt)

            example = {
                    'src': src, 
                    'tgt': tgt,
                    'len_src': len_src,
                    'len_tgt': len_tgt,
                    #'str_src': str_src,
                    #'str_tgt': str_tgt
                    }
            examples.append(example)
            dataloader = torch.utils.data.DataLoader(DictDataset(examples))
            dataloaders.append(dataloader)
    
    train, dev, test = dataloaders
    
    return train, dev, test







