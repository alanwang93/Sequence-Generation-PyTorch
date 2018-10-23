#!/usr/bin/env python
# coding=utf-8

from s2s.datasets import *

class Config:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.batch_size = 64


class Test(Config):

    def __init__(self, data_path, model_path):
        super().__init__(data_path, model_path)
        self.dataset = TestData(data_path)

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
    
    def __init__(self, data_path, model_path):
        super().__init__(data_path, model_path)
        self.dataset = Gigaword(data_path)
