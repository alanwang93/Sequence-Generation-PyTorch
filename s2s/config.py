#!/usr/bin/env python
# coding=utf-8

from s2s.datasets import *

class Config:
    def __init__(self, raw_root, data_root, model_path):
        self.raw_root = raw_root
        self.data_root = data_root
        self.model_path = model_path
        self.batch_size = 64
        self.max_step = 100000

        self.optimizer = 'Adam'
        self.optimizer_kwargs = dict(
                lr=0.001, 
                betas=(0.9, 0.999), 
                eps=1e-08, 
                weight_decay=0)
        self.clip_norm = 5.

        # log
        self.log_freq = 100
        self.eval_freq = 2000

        # validation
        self.metric = 'loss'
        self.init_metric = float('inf')

        # pretrained embedding
        # stored in raw_root/embeddings/...
        self.pretrained = None #'glove.6B.300d.txt'


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
        self.embed_dropout = 0.2
        self.rnn_dropout = 0.1
        self.mlp_dropout = 0.3
        # embedding
        self.pretrained = None
        self.pretrained_size = None
        self.projection = False
        self.lm_coef = 0.5


class Vanilla_GigawordSmall(Config):
    def __init__(self, raw_root, data_root, model_path):
        super().__init__(raw_root, data_root, model_path)
        self.model = 'Seq2seq'
        self.dataset = GigawordSmall(raw_root, data_root)
        self.hidden_size = 256
        self.num_layers = 2
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.0
        self.rnn_dropout = 0.0
        self.mlp_dropout = 0.2
        self.attn_type = 'bilinear'
        # embedding
        self.pretrained = None #'glove.6B.300d.txt'
        self.pretrained_size = None #300
        self.projection = None
        #self.clf_coef = 3.

        self.optimizer = 'Adam'
        self.optimizer_kwargs = dict(
                lr=0.001, 
                betas=(0.9, 0.999), 
                eps=1e-08, 
                weight_decay=0)
        self.clip_norm = 5.

        # log
        self.log_freq = 100
        self.eval_freq = 500



class BiClfInternal(Config):
    def __init__(self, raw_root, data_root, model_path):
        super().__init__(raw_root, data_root, model_path)
        self.model = 'BiClfSeq2seq'
        self.dataset = BiClfInternalData(raw_root, data_root)
        self.hidden_size = 200
        self.num_layers = 2
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.2
        self.rnn_dropout = 0.1
        self.mlp_dropout = 0.3
        # embedding
        self.pretrained = 'glove.6B.300d.txt'
        self.pretrained_size = 300
        self.projection = None
        self.clf_coef = 3.
        self.mlp1_size = 400


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
        self.mlp_dropout = 0.3
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
        self.mlp_dropout = 0.3
        # embedding
        self.pretrained = None
        self.pretrained_size = None
        self.projection = False

class MTS2S_Gigaword(Config):
    
    def __init__(self, raw_root, data_root, model_path):
        super().__init__(raw_root, data_root, model_path)
        self.model = 'BiClfSeq2seq'
        self.dataset = BiClfGigaword(raw_root, data_root)
        self.hidden_size = 512
        self.num_layers = 1
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.2
        self.rnn_dropout = 0.1
        self.mlp_dropout = 0.2
        # embedding
        self.pretrained = 'glove.6B.300d.txt'
        self.pretrained_size = 300
        self.projection = None
        self.clf_coef = 3.
        self.mlp1_size = 400

class MTS2SExt_Gigaword(Config):
    
    def __init__(self, raw_root, data_root, model_path):
        super().__init__(raw_root, data_root, model_path)
        self.model = 'BiClfSeq2seqExt'
        self.dataset = BiClfGigaword(raw_root, data_root)
        self.hidden_size = 512
        self.num_layers = 2
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.2
        self.rnn_dropout = 0.1
        self.mlp_dropout = 0.2
        # embedding
        self.pretrained = 'glove.6B.300d.txt'
        self.pretrained_size = 300
        self.projection = None
        self.clf_coef = 1.
        self.mlp1_size = None

class SelfFusion_Gigaword(Config):
    def __init__(self, raw_root, data_root, model_path):
        super().__init__(raw_root, data_root, model_path)
        self.model = 'SelfFusionSeq2seq'
        self.dataset = Gigaword(raw_root, data_root)
        self.hidden_size = 512
        self.num_layers = 1
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.2
        self.rnn_dropout = 0.1
        self.mlp_dropout = 0.2
        # embedding
        self.pretrained = 'glove.6B.300d.txt'
        self.pretrained_size = 300
        self.projection = None
        #self.clf_coef = 3.
        self.mlp1_size = 400

