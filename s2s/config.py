#!/usr/bin/env python
# coding=utf-8

from s2s.datasets import *

class Config:
    def __init__(self, data_root):
        self.data_root = data_root
        self.batch_size = 64


class Test(Config):

    def __init__(self, data_root):
        super().__init__(data_root)
        self.dataset = TestData(data_root)

        self.hidden_size = 100
        self.num_layers = 1
        self.embed_size = 200
        self.bidirectional = True
        self.dropout = 0.2
        self.rnn_dropout = 0
        self.embed_file = None

class Seq2seq_Gigaword(Config):
    
    def __init__(self, data_root):
        super().__init__(data_root)
        self.dataset = Gigaword(data_root)
