#!/usr/bin/env python
# coding=utf-8

import os

__all__ = ['Gigawords', 'TestData']

class DataConfig:
    
    def __init__(self, root):
        # default data root
        self.root = root

        # suffix
        self.src = 'src'
        self.tgt = 'tgt'
        
        self.unk_token = '<UNK>'
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'

        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3

        # sentence
        self.max_len_src = 100
        self.max_len_tgt = 20
        
        # Vocabulary
        self.share_vocab = False
        self.max_vocab_src = 200000
        self.max_vocab_tgt = 100000
        self.min_freq = 3
        self.lower = True


    def build_generator(self, src_file, tgt_file, vocab):
        def gen():
            with open(src_file, 'r') as s, open(tgt_file, 'r') as t:
                for src, tgt in zip(s, t):
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
                            'str_src': str_src,
                            'str_tgt': str_tgt
                            }
                    yield example
        return gen
    

    def build_corpus(self, fname):
        """
        Build sklearn-style corpus
        """
        corpus = []
        with open(fname, 'r') as f:
            for line in f:
                line = line.strip()
                if self.lower:
                    line = line.lower()
                corpus.append(line.split())
        return corpus


class Gigawords(DataConfig):

    def __init__(self, root):
        super().__init__(root)
        
        self.name = 'Gigawords'
        self.path = os.path.join(self.root, 'giga')

        self.train_prefix = os.path.join(self.path, 'train') 
        self.dev_prefix = os.path.join(self.path, 'dev') 
        self.test_prefix = os.path.join(self.path, 'test') 
        
        # sentence
        self.max_src_len = 100
        self.max_tgt_len = 20
        
        # Vocabulary
        self.share_vocab = False
        self.max_src_vocab = 200000
        self.max_tgt_vocab = 100000
        self.min_freq = 3
        self.lower = True


class TestData(DataConfig):
    def __init__(self, root):
        super().__init__(root)

        self.path = os.path.join(self.root, 'test')
  
        self.train_prefix = os.path.join(self.path, 'train') 
        self.dev_prefix = os.path.join(self.path, 'dev') 
        self.test_prefix = os.path.join(self.path, 'test')    

        # sentence
        self.max_src_len = 100
        self.max_tgt_len = 20
        
        # Vocabulary
        self.share_vocab = False
        self.max_src_vocab = 200000
        self.max_tgt_vocab = 100000
        self.min_freq = 3
        self.lower = True    

        self.unk_token = 'unk'
  



















