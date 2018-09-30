#!/usr/bin/env python
# coding=utf-8


from collections import Counter, defaultdict
import os
import itertools
import pickle
import numpy as np

def unk_idx():
    return UNK_IDX

class Vocab:

    """
    Vocabulary class
    """
    def __init__(self, data_config):
        """
        Args:
            data_config
        """
        self.data_config = data_config
        self.path = data_config.path
        self.pad_token = data_config.pad_token
        self.unk_token = data_config.unk_token
        self.sos_token = data_config.sos_token
        self.eos_token = data_config.eos_token
        self.pad_idx=0, self.unk_idx=1, self.sos_idx=2, self.eos_idx=3
        self.tokens = self.freqs = self.itos = self.stoi = self.vectors =  None
        self.keys = ['tokens', 'freqs', 'itos', 'stoi', 'unk_token', \
                'pad_token', 'sos_token', 'eos_token', 'vectors', \
                'data_config', 'path', 'pad_idx', 'unk_idx', 'sos_idx', \
                'eos_idx']

    def build(self, data=None, rebuild=True):
        """
        Args:
            data (list of token lists, or None)
            rebuild (boolean): Rebuild vocabulary
        """
        dc = self.data_config
        if not rebuild and os.path.exists(os.path.join(dc.path, 'vocab.pkl')):
            print("Loading vocab")
            self._load()
        elif data is not None:
            
            self._init_vocab()
            print("Building vocab")
            
            # Build vocab
            self.update(data)

            # Reference: torchtext
            min_freq = max(dc.min_freq, 1)
            max_vocab = None if dc.max_vocab is None else dc.max_vocab + len(self.itos)
            words_and_frequencies = list(self.freqs.items())
            words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True) # sort by order

            for word, freq in words_and_frequencies:
                if not (freq < min_freq or len(self.itos) == max_vocab):
                    self.itos.append(word)
 
            self.stoi.update({tok: i for i, tok in enumerate(self.itos)})
            self._dump()


    def _init_vocab(self):
        """
        Initialize vocabulary
        """
        specials = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        self.itos = list(specials)
        self.stoi = defaultdict(unk_idx)
        self.freqs = Counter([])

    def update(self, data):
        """
        Update current word frequency
        """
        allwords = list(itertools.chain.from_iterable(data))
        self.freqs.update(allwords)

    
    def _finish(self):
        """
        Finish updating and rebuild vocabulary
        """
        dc = self.data_config
        min_freq = max(dc.min_freq, 1)
        max_vocab = None if dc.max_vocab is None else dc.max_vocab + len(self.itos)
        words_and_frequencies = list(self.freqs.items())
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True) # sort by order

        for word, freq in words_and_frequencies:
            if not (freq < min_freq or len(self.itos) == max_vocab):
                self.itos.append(word)
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})
        self._dump()


    def summary(self):
        s = "Vocabulary summary:\n"
        s += '  Vocab size:\t{0}\n'.format(len(self.itos))
        s += '  Min freq:\t{0}\n'.format(self.data_config.min_freq)
        s += '  Max vocab:\t{0}\n'.format(self.data_config.max_vocab)
        print(s)
        with open(os.path.join(self.path, 'vocab_summary.txt'), 'w') as f:
            f.write(s)

        with open(os.path.join(self.path, 'vocab.txt'), 'w') as f:
            for w in self.itos:
                f.write(w + '\n')


    def __len__(self):
        return len(self.itos)


    def _dump(self):
        d = dict()
        for k in self.keys:
            d[k] = getattr(self, k)
        with open(os.path.join(self.path, 'vocab.pkl'), 'wb') as f:
            pickle.dump(d, f)


    def _load(self):
        with open(os.path.join(self.path, 'vocab.pkl'), 'rb') as f:
            d = pickle.load(f)
        for k in self.keys:
            setattr(self, k, d[k])


    def toi(self, l):
        # TODO: it's wrong, because str has __len__()
        if hasattr(l, '__len__') and type(l) is not str:
            return [self.stoi[w] for w in l]
        else:
            return self.stoi[l]


    def tos(self, l):
        if hasattr(l, '__len__'):
            return [self.itos[w] for w in l]
        else:
            return self.itos[l]




