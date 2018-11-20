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

        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.5
        self.rnn_dropout = 0.5
        self.mlp_dropout = 0.5
        self.attn_type = 'concat'
        self.input_feed = False
        # embedding
        self.pretrained = 'glove.6B.300d.txt'
        self.pretrained_size = 300
        self.projection = None

        self.optimizer = 'Adam'
        self.optimizer_kwargs = dict(
                lr=0.001, 
                betas=(0.9, 0.999), 
                eps=1e-06, 
                weight_decay=0)

        self.clip_norm = 5.
        self.patience = 5 # decay lr if dev metric doesn't improve for n steps
        self.decay_factor = 0.5
        self.max_weight_value = 15

        # log
        self.log_freq = 100
        self.eval_freq = 1000



        # validation
        self.metric = 'ROUGE-1'
        self.init_metric = 0.
        self.max_length = 50
        self.input_feed = False

        self.scheduler_mode = 'max'

    def is_better(self, cur, best):
        if cur > best:
            return True
        return False


class Vanilla(Config):
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)
        self.model = 'Seq2seq'
        self.encoder = 'RNNEncoder'
        self.decoder = 'AttnRNNDecoder'
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.5
        self.rnn_dropout = 0.5
        #self.mlp_dropout = 0.5
        self.attn_type = 'concat'
        self.input_feed = False
        # embedding
        self.pretrained = 'glove.6B.300d.txt'
        self.pretrained_size = 300
        self.projection = None

        self.optimizer = 'Adam'
        self.clip_norm = 5.
        self.patience = 5 # decay lr if dev metric doesn't improve for n steps
        self.decay_factor = 0.5

        # log
        self.log_freq = 100
        self.eval_freq = 1000


class SEASS(Config):
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)
        self.model = 'Seq2seq'
        self.encoder = 'SelectiveEncoder'
        self.decoder = 'AttnRNNDecoder'
        self.enc_hidden_size = 256
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.5
        self.rnn_dropout = 0.5
        self.mlp_dropout = 0.5
        self.attn_type = 'concat'
        # embedding
        self.pretrained = 'glove.6B.300d.txt'
        self.pretrained_size = 300
        self.projection = None

        self.optimizer = 'Adam'
        self.clip_norm = 5.
        self.patience = 5 # decay lr if dev metric doesn't improve for n steps
        self.decay_factor = 0.5

        # log
        self.log_freq = 100
        self.eval_freq = 1000



class PosFeedS2S(Config):
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)
        self.model = 'Seq2seq'
        self.encoder = 'PosFeedEncoder'
        self.decoder = 'PosFeedDecoder'
        self.hidden_size = 512
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.5
        self.rnn_dropout = 0.5
        self.mlp_dropout = 0.5
        self.attn_type = 'concat'
        # embedding
        self.pretrained = 'glove.6B.300d.txt'
        self.pretrained_size = 300
        self.projection = None

        self.optimizer = 'Adam'
        self.clip_norm = 5.
        self.patience = 5 # halve lr if dev metric doesn't improve for n steps
        self.decay_factor = 0.5

        # log
        self.log_freq = 100
        self.eval_freq = 1000

class PosGatedS2S(Config):
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)
        self.model = 'PosGatedSeq2seq'
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
        self.pretrained_size = 300
        self.projection = None

        self.optimizer = 'Adam'
        self.clip_norm = 5.
        self.patience = 5 # halve lr if dev metric doesn't improve for n steps
        self.decay_factor = 0.5

        # log
        self.log_freq = 100
        self.eval_freq = 1000

class AttGateS2S(Config):
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)
        self.model = 'AttGateSeq2seq'
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
        self.clip_norm = 5.
        self.patience = 5 # halve lr if dev metric doesn't improve for n steps
        self.decay_factor = 0.5

        # log
        self.log_freq = 100
        self.eval_freq = 1000

class AGS2S(Config):
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)
        self.model = 'Seq2seq'
        self.encoder = 'RNNEncoder'
        self.decoder = 'AGDecoder'
        self.hidden_size = 512
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.5
        self.rnn_dropout = 0.5
        self.mlp_dropout = 0.5
        self.attn_type = 'concat'
        # embedding
        self.pretrained = 'glove.6B.300d.txt'
        self.pretrained_size = 300
        self.projection = None

        self.optimizer = 'Adam'
        self.clip_norm = 5.
        self.patience = 5 # halve lr if dev metric doesn't improve for n steps
        self.decay_factor = 0.5

        # log
        self.log_freq = 100
        self.eval_freq = 1000

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
        self.clip_norm = 5.
        self.patience = 5 # halve lr if dev metric doesn't improve for n steps
        self.decay_factor = 0.5

        # log
        self.log_freq = 100
        self.eval_freq = 1000


class VanillaIF(Config):
    """
    With input feed (add context vector to input)
    """
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
        self.clip_norm = 5.
        self.patience = 5 # decay lr if dev metric doesn't improve for n steps
        self.decay_factor = 0.5

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
        self.clip_norm = 5.

        # log
        self.log_freq = 100
        self.eval_freq = 500



class MTS2S_v2(Config):
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)

        self.model = 'MTSeq2seq_v2'
        self.hidden_size = 512
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.0
        self.rnn_dropout = 0.5
        self.mlp_dropout = 0.5
        # embedding
        self.pretrained = 'glove.6B.300d.txt'
        self.pretrained_size = 300 #300
        self.projection = False
        self.mt_coef = 0.1

        self.attn_type = 'concat'
        #self.mlp1_size = None

        self.optimizer = 'Adam'
        self.clip_norm = 5.
        self.patience = 5 # halve lr if dev metric doesn't improve for n steps
        # log
        self.log_freq = 100
        self.eval_freq = 1000


class MTS2S(Config):
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)

        self.model = 'MTSeq2seq'
        self.hidden_size = 512
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.5
        self.rnn_dropout = 0.5
        self.mlp_dropout = 0.5
        # embedding
        self.pretrained = 'glove.6B.300d.txt'
        self.pretrained_size = 300
        self.projection = False
        self.mt_coef = 2.

        self.attn_type = 'concat'
        #self.mlp1_size = None

        self.optimizer = 'Adam'
        self.clip_norm = 5.
        self.patience = 5 # halve lr if dev metric doesn't improve for n steps
        # log
        self.log_freq = 100
        self.eval_freq = 1000


class GatedMTS2S(Config):
    
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)
        self.model = 'GatedMTSeq2seq'
        self.hidden_size = 512
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.embed_size = 300
        self.bidirectional = True
        self.embed_dropout = 0.0
        self.rnn_dropout = 0.5
        self.mlp_dropout = 0.5
        # embedding
        self.pretrained = 'glove.6B.300d.txt'
        self.pretrained_size = 300
        self.projection = False
        self.mt_coef = 0.1

        self.attn_type = 'concat'

        self.optimizer = 'Adam'
        self.clip_norm = 5.
        self.patience = 5 # halve lr if dev metric doesn't improve for n steps
        #self.decay_factor = 0.5
        
        # log
        self.log_freq = 100
        self.eval_freq = 1000


class Fusion(Config):
    def __init__(self, raw_root, data_root, model_path, dataset):
        super().__init__(raw_root, data_root, model_path, dataset)
        self.model = 'Seq2seq'
        self.encoder = 'FusionEncoder'
        self.decoder = 'AttnRNNDecoder'
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
        self.clip_norm = 5.
        self.patience = 5 # decay lr if dev metric doesn't improve for n steps
        self.decay_factor = 0.5

        # log
        self.log_freq = 100
        self.eval_freq = 1000

