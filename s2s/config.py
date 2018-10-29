#!/usr/bin/env python
# coding=utf-8

from s2s.datasets import *

class Config:
    def __init__(self, data_root, model_path):
        self.data_root = data_root
        self.model_path = model_path
        self.batch_size = 64
        self.max_step = 20000

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
        self.eval_metric = 'loss'
        self.init_metric = float('inf')
        
    def is_better(cur, best):
        if cur < best:
            return True
        return False


class Test(Config):

    def __init__(self, data_root, model_path):
        super().__init__(data_root, model_path)
        self.model = 'Seq2seq'
        self.dataset = TestData(data_root)
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
    
    def __init__(self, data_root, model_path):
        super().__init__(data_root, model_path)
        self.dataset = Gigaword(data_root)
