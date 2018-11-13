#!/usr/bin/env python
# coding=utf-8

import s2s.datasets as datasets

class Config:
    def __init__(self, raw_root, data_root, model_path, dataset):
        self.raw_root = raw_root
        self.data_root = data_root
        self.model_path = model_path
        self.dataset = getattr(datasets, dataset)(raw_root, data_root)
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
        self.metric = 'ROUGE-2'
        self.init_metric = 0.

        # pretrained embedding
        # stored in raw_root/embeddings/...
        self.pretrained = None #'glove.6B.300d.txt'

        self.scheduler_mode = 'max'
    def is_better(self, cur, best):
        if cur > best:
            return True
        return False


class GatedDecoder(Config):
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)
        self.model = 'GatedSeq2seq'
        self.hidden_size = 512
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.0
        self.rnn_dropout = 0.5
        self.mlp_dropout = 0.5
        self.attn_type = 'concat'
        # embedding
        self.pretrained = None #'glove.6B.300d.txt'
        self.pretrained_size = None #300
        self.projection = None

        self.optimizer = 'Adam'
        self.optimizer_kwargs = dict(
                lr=0.001, 
                betas=(0.9, 0.999), 
                eps=1e-08, 
                weight_decay=0)
        self.clip_norm = 5.
        self.patience = 5 # halve lr if dev metric doesn't improve for n steps

        # log
        self.log_freq = 100
        self.eval_freq = 1000


class Vanilla(Config):
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)
        self.model = 'Seq2seq'
        self.hidden_size = 512
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.0
        self.rnn_dropout = 0.5
        self.mlp_dropout = 0.5
        self.attn_type = 'concat'
        # embedding
        self.pretrained = 'glove.6B.300d.txt'
        self.pretrained_size = 300
        self.projection = None

        self.optimizer = 'Adam'
        self.optimizer_kwargs = dict(
                lr=0.001, 
                betas=(0.9, 0.999), 
                eps=1e-08, 
                weight_decay=0)
        self.clip_norm = 5.
        self.patience = 5 # halve lr if dev metric doesn't improve for n steps

        # log
        self.log_freq = 100
        self.eval_freq = 1000

class VanillaMedium(Config):
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)

        self.model = 'Seq2seq'
        #self.dataset = GigawordSmall(raw_root, data_root)
        self.hidden_size = 256
        self.num_layers = 2
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.0
        self.rnn_dropout = 0.0
        self.mlp_dropout = 0.2
        self.attn_type = 'concat'
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


class MTS2S(Config):
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)

        self.model = 'MTSeq2seq'
        self.hidden_size = 512
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.0
        self.rnn_dropout = 0.5
        self.mlp_dropout = 0.5
        # embedding
        self.pretrained = None #'glove.6B.300d.txt'
        self.pretrained_size = None #300
        self.projection = False
        self.mt_coef = 0.1

        self.attn_type = 'concat'
        #self.mlp1_size = None

        self.optimizer = 'Adam'
        self.optimizer_kwargs = dict(
                lr=0.001, 
                betas=(0.9, 0.999), 
                eps=1e-08, 
                weight_decay=0)
        self.clip_norm = 5.
        self.patience = 5 # halve lr if dev metric doesn't improve for n steps
        # log
        self.log_freq = 100
        self.eval_freq = 1000


class MTS2SExt(Config):
    
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)
        self.model = 'MTSeq2seqExt'
        self.hidden_size = 256
        self.num_layers = 2
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.0
        self.rnn_dropout = 0.0
        self.mlp_dropout = 0.2
        # embedding
        self.pretrained = None #'glove.6B.300d.txt'
        self.pretrained_size = None #300
        self.projection = False
        self.clf_coef = 1.

        self.attn_type = 'symmetric'

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

