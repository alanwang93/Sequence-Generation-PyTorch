#!/usr/bin/env python
# coding=utf-8

from s2s.datasets import *

class Config:
    def __init__(self):
        self.batch_size = 3


class Test(Config):

    def __init__(self):
        super().__init__()
        self.dataset = TestData('/Users/wangyifan/workspace/research/sequence-to-sequence/data')


class Seq2seq_Gigaword(Config):
    
    def __init__(self):
        super().__init__()
        self.dataset = Gigaword()
