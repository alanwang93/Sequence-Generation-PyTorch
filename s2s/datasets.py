#!/usr/bin/env python
# coding=utf-8

import os

__all__ = ['Gigaword', 'BiClfGigaword', 'TestData', 'TestData2', 'BiClfTestData', 
        'BiClfInternalData']

class DataConfig:
    
    def __init__(self, raw_root, data_root):
        # default data root
        self.raw_root = raw_root
        self.data_root = data_root

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
        self.src_out = False
        
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


class Gigaword(DataConfig):
    """
    use `train.py`
    """

    def __init__(self, raw_root, data_root):
        super().__init__(raw_root, data_root)
        
        self.name = 'Gigawords'
        self.raw = os.path.join(self.raw_root, 'giga')
        self.path = os.path.join(self.data_root, 'giga')

        self.train_prefix = 'train'
        self.dev_prefix = 'dev'
        self.test_prefix = 'test.filtered'
        
        # sentence
        self.max_src_len = 120
        self.max_tgt_len = 50
        self.src_out = False
        
        # Vocabulary
        self.share_vocab = False
        self.max_src_vocab = 200000
        self.max_tgt_vocab = 100000
        self.min_freq = 0
        self.lower = True

        self.unk_token = '<unk>'

class BiClfGigaword(DataConfig):
    """
    use `train_biclf.py`
    """

    def __init__(self, raw_root, data_root):
        super().__init__(raw_root, data_root)
        
        self.name = 'BiClfGigawords'
        self.raw = os.path.join(self.raw_root, 'giga')
        self.path = os.path.join(self.data_root, 'biclf_giga')

        self.train_prefix = 'train'
        self.dev_prefix = 'dev'
        self.test_prefix = 'test.filtered'
        
        # sentence
        self.max_src_len = 120
        self.max_tgt_len = 50
        self.src_out = True
        
        # Vocabulary
        self.share_vocab = False
        self.max_src_vocab = 200000
        self.max_tgt_vocab = 100000
        self.min_freq = 0
        self.lower = True

        self.unk_token = '<unk>'

class TestData(DataConfig):
    def __init__(self, raw_root, data_root):
        super().__init__(raw_root, data_root)

        self.raw = os.path.join(self.raw_root, 'test')
        self.path = os.path.join(self.data_root, 'test')
  
        self.train_prefix = 'train'
        self.dev_prefix = 'dev'
        self.test_prefix = 'test'

        self.src_out = True

        # sentence
        self.max_src_len = 100
        self.max_tgt_len = 50
        
        # Vocabulary
        self.share_vocab = False
        self.max_src_vocab = 200000
        self.max_tgt_vocab = 100000
        self.min_freq = 0
        self.lower = True    

        self.unk_token = 'unk'
  

class TestData2(DataConfig):
    def __init__(self, raw_root, data_root):
        super().__init__(raw_root, data_root)

        self.raw = os.path.join(self.raw_root, 'test')
        self.path = os.path.join(self.data_root, 'test2')
  
        self.train_prefix = 'train'
        self.dev_prefix = 'dev'
        self.test_prefix = 'test'

        self.src_out = False

        # sentence
        self.max_src_len = 100
        self.max_tgt_len = 20
        
        # Vocabulary
        self.share_vocab = False
        self.max_src_vocab = 200000
        self.max_tgt_vocab = 100000
        self.min_freq = 1
        self.lower = True    

        self.unk_token = 'unk'
  

class BiClfTestData(DataConfig):
    def __init__(self, raw_root, data_root):
        super().__init__(raw_root, data_root)

        self.raw = os.path.join(self.raw_root, 'test')
        self.path = os.path.join(self.data_root, 'biclf_test')
  
        self.train_prefix = 'train'
        self.dev_prefix = 'dev'
        self.test_prefix = 'test.filtered'

        # sentence
        self.max_src_len = 100
        self.max_tgt_len = 20
        
        # Vocabulary
        self.share_vocab = False
        self.max_src_vocab = 200000
        self.max_tgt_vocab = 100000
        self.min_freq = 0
        self.lower = True    

        self.unk_token = '<unk>'




class BiClfInternalData(DataConfig):
    def __init__(self, raw_root, data_root):
        super().__init__(raw_root, data_root)

        self.raw = os.path.join(self.raw_root, 'internal')
        self.path = os.path.join(self.data_root, 'biclf_internal')
  
        self.train_prefix = 'train'
        self.dev_prefix = 'dev'
        self.test_prefix = 'test.filtered'

        # sentence
        self.max_src_len = 100
        self.max_tgt_len = 20
        
        # Vocabulary
        self.share_vocab = False
        self.max_src_vocab = 200000
        self.max_tgt_vocab = 100000
        self.min_freq = 0
        self.lower = True    

        self.unk_token = '<unk>'













