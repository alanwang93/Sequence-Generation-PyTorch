#!/usr/bin/env python
# coding=utf-8

from s2s.datasets import *

class Config:
    def __init__(self, raw_root, data_root, model_path):
        self.data_root = data_root
        self.model_path = model_path
        self.batch_size = 32
        self.max_step = 30000

        self.optimizer = 'Adam'
        self.optimizer_kwargs = dict(
                lr=0.001, 
                betas=(0.9, 0.999), 
                eps=1e-08, 
                weight_decay=0)

        # log
        self.log_freq = 10
        self.eval_freq = 100

        # validation
        self.metric = 'loss'
        self.init_metric = float('inf')
        
    def is_better(self, cur, best):
        if cur < best:
            return True
        return False


class EnLM(Config):
    def __init__(self, raw_root, data_root, model_path):
        super().__init__(raw_root, data_root, model_path)
        self.model = 'EnLMSeq2seq'
        self.dataset = TestData(raw_root, data_root)
        self.hidden_size = 100
        self.num_layers = 1
        self.embed_size = 200
        self.bidirectional = True
        self.dropout = 0.2
        self.rnn_dropout = 0
        # embedding
        self.pretrained = None
        self.pretrained_size = None
        self.projection = False
        self.lm_coef = 0.5



class BiClf(Config):
    def __init__(self, raw_root, data_root, model_path):
        super().__init__(raw_root, data_root, model_path)
        self.model = 'BiClfSeq2seq'
        self.dataset = BiClfTestData(raw_root, data_root)
        self.hidden_size = 200
        self.num_layers = 2
        self.embed_size = 300
        self.bidirectional = True
        self.dropout = 0.3
        self.rnn_dropout = 0.2
        # embedding
        self.pretrained = None
        self.pretrained_size = None
        self.projection = False
        self.clf_coef = 1.


class Test(Config):

    def __init__(self, raw_root, data_root, model_path):
        super().__init__(raw_root, data_root, model_path)
        self.model = 'Seq2seq'
        self.dataset = TestData(raw_root, data_root)
        self.hidden_size = 100
        self.num_layers = 1
        self.embed_size = 200
        self.bidirectional = True
        self.dropout = 0.2
        self.rnn_dropout = 0
        # embedding
        self.pretrained = None
        self.pretrained_size = None
        self.projection = False

class Test2(Config):

    def __init__(self, raw_root, data_root, model_path):
        super().__init__(raw_root, data_root, model_path)
        self.model = 'Seq2seq'
        self.dataset = TestData2(raw_root, data_root)
        self.hidden_size = 100
        self.num_layers = 1
        self.embed_size = 200
        self.bidirectional = True
        self.dropout = 0.2
        self.rnn_dropout = 0
        # embedding
        self.pretrained = None
        self.pretrained_size = None
        self.projection = False

class Seq2seq_Gigaword(Config):
    
    def __init__(self, raw_root, data_root, model_path):
        super().__init__(raw_root, data_root, model_path)
        self.dataset = Gigaword(raw_root, data_root)
