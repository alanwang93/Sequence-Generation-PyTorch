#!/usr/bin/env python
# coding=utf-8


import os

class DataConfig:

    def __init__(self):
        # default data root
        self.root = None
        
        # suffix
        self.src = 'src'
        self.tgt = 'tgt'
        
        self.unk_token = '<UNK>'
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        

class Gigawords(DataConfig):

    def __init__(self):
        super().__init__()
        self.name = 'Gigawords'
        self.path = os.path.join(self.root, 'giga')
        self.train_prefix = [os.path.join(self.path, 'train.article.txt')]
        self.train_tgt = [os.path.join(self.path, 'train.title.txt')]
        self.valid_src = [os.path.join(self.path, 'valid.article.txt')]
        self.valid_tgt = [os.path.join(self.path, 'valid.title.txt')]
        self.test_src = [os.path.join(self.path, 'test.article.txt')]
        self.test_tgt = [os.path.join(self.path, 'test.title.txt')]

        self.max_src_len = 100
        self.max_tgt_len = 20

        self.max_vocab = 200000
        self.min_freq = 3
    
    def build_dataset(self, src_file, tgt_file, vocab):
        class GigawordsDataset(Dataset):
    



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
    
    

    def build_corpus(self, src_file, tgt_file):
        corpus = []
        with open(src_file, 'r') as s, open(tgt_file, 'r') as t:
            for src, tgt in zip(s, t):
                src = src.strip().lower()
                tgt = tgt.strip().lower()
                corpus.append(src.split())
                corpus.append(tgt.split())
        return corpus

















