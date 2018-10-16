#!/usr/bin/env python
# coding=utf-8
import os
import time

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
    # Default pad idx == 0

    keys = ['src', 'tgt_in', 'tgt_out', 'len_src', 'len_tgt']
    data = {}
    for k in keys:
        data[k] = []
    for d in batch:
        for k in keys:
            data[k].append(d[k])
    for k in keys:
        data[k] = torch.from_numpy(np.asarray(data[k]))
    #data['src'] = pad_sequence(data['src'], True, 0)
    #data['tgt_in'] = pad_sequence(data['tgt_in'], True, 0)
    #data['tgt_out'] = pad_sequence(data['tgt_out'], True, 0)
    #data['src'] = torch.stack(data['src'])
    #data['tgt_in'] = torch.stack(data['tgt_in'])
    #data['tgt_out'] = torch.stack(data['tgt_out'])
    
    #data['len_src'] = torch.stack(data['len_src'])
    #data['len_tgt'] = torch.stack(data['len_tgt'])
    return data



def build_dataloaders(config, src_vocab, tgt_vocab):
    print('Building dataloaders...')
    cf = config
    dc = config.dataset # data config
    max_len_src = dc.max_len_src
    max_len_tgt = dc.max_len_tgt
    dataloaders = []
    for prefix in [dc.train_prefix,
                   dc.dev_prefix,
                   dc.test_prefix]:
        
        src_file = prefix + '.' + dc.src
        tgt_file = prefix + '.' + dc.tgt
        examples = []
        
        src_npz = src_file + '.npz'
        tgt_npz = tgt_file + '.npz'

        if os.path.exists(src_npz) and os.path.exists(tgt_npz):
            start = time.time()
            src_data = np.load(src_npz)
            all_src = src_data['src']
            all_len_src = src_data['len_src']
            tgt_data = np.load(tgt_npz)
            all_tgt = tgt_data['tgt']
            all_len_tgt = tgt_data['len_tgt']
            print('Loading takes {0:.3f} s'.format(time.time()-start))
        else:
            with open(src_file, 'r') as s, open(tgt_file, 'r') as t:
                start = time.time()

                all_len_src, all_len_tgt = [], []
                all_src, all_tgt = [], []
                i = 0
                for src, tgt in zip(s, t):
                    i += 1
                    if i % 1000 == 0:
                        print(i)
                    str_src = src.strip().lower().split(' ')[:max_len_src]
                    str_tgt = tgt.strip().lower().split(' ')[:max_len_tgt-1] # max_len_tgt == final idx tensor length
                    len_src = len(str_src)
                    len_tgt = len(str_tgt)+1 # including EOS or SOS
                    src = src_vocab.toi(str_src)
                    tgt = tgt_vocab.toi(str_tgt)
                    src = np.pad(src, (0, max_len_src-len_src), 'constant', constant_values=0)
                    tgt = np.pad(tgt, (0, max_len_tgt-len_tgt+1), 'constant', constant_values=0)

                    all_len_src.append(len_src)
                    all_len_tgt.append(len_tgt)
                    all_src.append(src)
                    all_tgt.append(tgt)
             
                print('Processing data takes {0:.3f} s'.format(time.time()-start))
                print('Saving processed data...')
                np.savez(src_npz, src=np.asarray(all_src), len_src=np.asarray(all_len_src))
                np.savez(tgt_npz, tgt=np.asarray(all_tgt), len_tgt=np.asarray(all_len_tgt))
                
                
        for i in range(len(all_len_src)):
            tgt = all_tgt[i]
            len_tgt = all_len_tgt[i]
            len_src = all_len_src[i]
            tgt_in = np.concatenate(([dc.sos_idx], tgt[:-1]))
            tgt[len_tgt-1] = dc.eos_idx
            example = {
                    'src': all_src[i], 
                    'tgt_in': tgt_in,
                    'tgt_out': tgt,
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







