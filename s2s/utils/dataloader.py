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
    # TODO: keys from batch
    # keys = ['src', 'tgt_in', 'tgt_out', 'len_src', 'len_tgt']
    keys = list(batch[0].keys())
    data = {}
    for k in keys:
        data[k] = []
    for d in batch:
        for k in keys:
            data[k].append(d[k])
    for k in keys:
        data[k] = torch.from_numpy(np.asarray(data[k]))
    return data


def build_dataloaders(
        config, 
        src_vocab, 
        tgt_vocab, 
        rebuild=False):
    """
    src_in:
        if src_out == False: t0, ... tn
        else: SOS, t0, ... tn
    len_src: t0, ... tn, EOS
    tgt_in: SOS, t0, .. tn
    tgt_out: t0, ... tn, EOS
    len_tgt:
    """
    cf = config
    dc = config.dataset # data config
    src_out = dc.src_out
    max_len_src = dc.max_len_src
    max_len_tgt = dc.max_len_tgt
    dataloaders = []
    for prefix in [dc.train_prefix,
                   dc.dev_prefix,
                   dc.test_prefix]:
        
        prefix_file = os.path.join(dc.raw, prefix)
        src_file = prefix_file + '.' + dc.src
        tgt_file = prefix_file + '.' + dc.tgt
        examples = []
        
        prefix_npz = os.path.join(dc.path, prefix)
        src_npz =  prefix_npz + '.' + dc.src + '.npz'
        tgt_npz = prefix_npz + '.' + dc.tgt + '.npz'

        if os.path.exists(src_npz) and os.path.exists(tgt_npz) and not rebuild:
            print('Loading dataloaders...')
            start = time.time()
            src_data = np.load(src_npz)
            all_src = src_data['src']
            all_len_src = src_data['len_src']
            tgt_data = np.load(tgt_npz)
            all_tgt = tgt_data['tgt']
            all_len_tgt = tgt_data['len_tgt']
            print('Loading takes {0:.3f} s'.format(time.time()-start))
        else:
            print('Building dataloaders...')
            with open(src_file, 'r') as s, open(tgt_file, 'r') as t:
                start = time.time()

                all_len_src, all_len_tgt = [], []
                all_src, all_tgt = [], []
                i = 0
                for src, tgt in zip(s, t):
                    i += 1
                    if i % 1000 == 0:
                        print(i, flush=True)
                    if src_out:
                        str_src = src.strip().lower().split(' ')[:max_len_src-1]
                        len_src = len(str_src)+1
                        src = src_vocab.toi(str_src)
                        src = np.pad(src, (0, max_len_src-len_src+1), 'constant', constant_values=0)
                    else:
                        str_src = src.strip().lower().split(' ')[:max_len_src]
                        len_src = len(str_src)
                        src = src_vocab.toi(str_src)
                        src = np.pad(src, (0, max_len_src-len_src), 'constant', constant_values=0)
                    
                    str_tgt = tgt.strip().lower().split(' ')[:max_len_tgt-1] # max_len_tgt == final idx tensor length
                    len_tgt = len(str_tgt)+1 # including EOS or SOS
                    tgt = tgt_vocab.toi(str_tgt)
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
            src = all_src[i]
            len_tgt = all_len_tgt[i]
            len_src = all_len_src[i]
            tgt_in = np.concatenate(([dc.sos_idx], tgt[:-1]))
            tgt[len_tgt-1] = dc.eos_idx
            if src_out:
                src_in = np.concatenate(([dc.sos_idx], src[:-1]))
                src[len_src-1] = dc.eos_idx
            else:
                src_in = src

            example = {
                    'src_in': src_in, 
                    'src_out': src, 
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


def build_dataloaders_biclf(
        config, 
        src_vocab, 
        tgt_vocab, 
        rebuild=False):
    """
    src_in:
        if src_out == False: t0, ... tn
        else: SOS, t0, ... tn
    len_src: t0, ... tn, EOS
    tgt_in: SOS, t0, .. tn
    tgt_out: t0, ... tn, EOS
    len_tgt:
    """
    cf = config
    dc = config.dataset # data config
    max_len_src = dc.max_len_src
    max_len_tgt = dc.max_len_tgt
    dataloaders = []
    for prefix in [dc.train_prefix,
                   dc.dev_prefix,
                   dc.test_prefix]:
        
        prefix_file = os.path.join(dc.raw, prefix)
        src_file = prefix_file + '.' + dc.src
        tgt_file = prefix_file + '.' + dc.tgt
        examples = []
        
        prefix_npz = os.path.join(dc.path, prefix)
        src_npz =  prefix_npz + '.' + dc.src + '.npz'
        tgt_npz = prefix_npz + '.' + dc.tgt + '.npz'

        if os.path.exists(src_npz) and os.path.exists(tgt_npz) and not rebuild:
            print('Loading dataloaders...')
            start = time.time()
            src_data = np.load(src_npz)
            all_src = src_data['src']
            all_len_src = src_data['len_src']
            tgt_data = np.load(tgt_npz)
            all_tgt = tgt_data['tgt']
            all_len_tgt = tgt_data['len_tgt']
            all_src_out = src_data['src_out']
            print('Loading takes {0:.3f} s'.format(time.time()-start))
        else:
            print('Building dataloaders...')
            with open(src_file, 'r') as s, open(tgt_file, 'r') as t:
                start = time.time()

                all_len_src, all_len_tgt = [], []
                all_src, all_tgt = [], []
                all_src_out = []
                k = 0
                for src, tgt in zip(s, t):
                    k += 1
                    if k % 1000 == 0:
                        print(k, flush=True)

                    str_src = src.strip().lower().split(' ')[:max_len_src]
                    len_src = len(str_src)
                    src = src_vocab.toi(str_src)
                    src = np.pad(src, (0, max_len_src-len_src), 'constant', constant_values=0)
                    
                    str_tgt = tgt.strip().lower().split(' ')[:max_len_tgt-1] # max_len_tgt == final idx tensor length
                    len_tgt = len(str_tgt)+1 # including EOS or SOS
                    tgt = tgt_vocab.toi(str_tgt)
                    tgt = np.pad(tgt, (0, max_len_tgt-len_tgt+1), 'constant', constant_values=0)

                    # Binary classification
                    # 1: src token in tgt string
                    # 2: no
                    src_out = [0]*max_len_src
                    for i in range(len_src):
                        if str_src[i] in str_tgt:
                            src_out[i] = 1
                        else:
                            src_out[i] = 2

                    all_len_src.append(len_src)
                    all_len_tgt.append(len_tgt)
                    all_src.append(src)
                    all_tgt.append(tgt)
                    all_src_out.append(src_out)
             
                print('Processing data takes {0:.3f} s'.format(time.time()-start))
                print('Saving processed data...')
                np.savez(src_npz, src=np.asarray(all_src), len_src=np.asarray(all_len_src), src_out=np.asarray(all_src_out))
                np.savez(tgt_npz, tgt=np.asarray(all_tgt), len_tgt=np.asarray(all_len_tgt))
                
                
        for i in range(len(all_len_src)):
            tgt = all_tgt[i]
            src_in = all_src[i]
            len_tgt = all_len_tgt[i]
            len_src = all_len_src[i]
            tgt_in = np.concatenate(([dc.sos_idx], tgt[:-1]))
            tgt[len_tgt-1] = dc.eos_idx
            src_out = all_src_out[i]

            example = {
                    'src_in': src_in, 
                    'src_out': src_out, 
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





