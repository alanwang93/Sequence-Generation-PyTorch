#!/usr/bin/env python
# coding=utf-8


import os
import time


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from s2s.utils.utils import get_vocab_weights
from s2s.utils.losses import WeightedCE, DecayCE
from s2s.models.common import Embedding, DecInit
from s2s.models.encoder import RNNEncoder, FusionEncoder, SelectiveEncoder, PosGatedEncoder
from s2s.models.decoder import RNNDecoder, AttnRNNDecoder, GatedAttnRNNDecoder, \
    AGDecoder, AttGateDecoder

from s2s.models import encoder
from s2s.models import decoder


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()


    def clip_weight(self, max_weight_value):
        if max_weight_value:
            for p in self.params:
                p.data.clamp_(0 - max_weight_value, max_weight_value)
 
class Seq2seq(BaseModel):
    def __init__(self, config, src_vocab, tgt_vocab, restore):
        super().__init__()
        cf = config
        dc = cf.dataset

        self.config = config
        self.sos_idx = dc.sos_idx
        self.eos_idx = dc.eos_idx
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        if cf.pretrained is not None:
            pretrained = os.path.join(cf.raw_root, 'embeddings', cf.pretrained)
        else:
            pretrained = None
        if restore is not None:
            pretrained = None

        encoder_embed = Embedding(
                cf.embed_size,
                src_vocab,
                pretrained,
                cf.pretrained_size,
                cf.projection,
                cf.embed_dropout)
        
        decoder_embed = Embedding(
                cf.embed_size,
                tgt_vocab,
                pretrained,
                cf.pretrained_size,
                cf.projection,
                cf.embed_dropout)

        self.encoder = getattr(encoder, cf.encoder)(
                cf.enc_hidden_size,
                cf.enc_num_layers,
                encoder_embed,
                cf.bidirectional,
                cf.rnn_dropout)

        self.decoder = getattr(decoder, cf.decoder)(
                cf.dec_hidden_size,
                cf.dec_num_layers,
                decoder_embed,
                cf.rnn_dropout,
                cf.mlp_dropout,
                cf.attn_type)

        self.dec_initer = DecInit(
                cf.enc_hidden_size, 
                cf.dec_hidden_size,
                False)# cf.bidirectional)

        self.params = list(self.parameters()) #list(self.encoder.parameters()) +  list(self.decoder.parameters())
        self.optimizer = getattr(torch.optim, cf.optimizer)(self.params, **cf.optimizer_kwargs)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                cf.scheduler_mode,
                factor=0.5,
                patience=cf.patience,
                verbose=True)
        self.weighted_loss = False
        self.loss = nn.CrossEntropyLoss(
                reduction='sum', 
                ignore_index=dc.pad_idx)

        if self.weighted_loss:
            self.loss = DecayCE(self.loss)

    def forward(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, tgt_out, len_tgt = batch['tgt_in'], batch['tgt_out'], batch['len_tgt']
        src_output, src_last = self.encoder(src, len_src)
        dec_init = self.dec_initer(src_last[1].unsqueeze(0))#.unsqueeze(0) # last back hidden
        logits = self.decoder(tgt_in, len_tgt, dec_init, src_output, len_src)
        return logits

    def train_step(self, batch):
        logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        if self.weighted_loss:
            loss = self.loss(logits, batch['tgt_out'])
        else:
            loss = self.loss(logits.view(batch_size*seq_len, -1), batch['tgt_out'].view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.config.clip_norm)
        self.optimizer.step()
        self.clip_weight(self.config.max_weight_value)
        return loss.item(), batch_size

    def get_loss(self, batch):
        logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        if self.weighted_loss:
            loss = self.loss(input=logits, target=batch['tgt_out'])
        else:
            loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))

        return loss.item(), batch_size  

    def greedy_decode(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, len_tgt = batch['tgt_in'],  batch['len_tgt']
        src_outputs, src_last = self.encoder(src, len_src)
        dec_init = self.dec_initer(src_last[1].unsqueeze(0)) #.unsqueeze(0) # last back hidden
        preds = self.decoder.greedy_decode(dec_init, self.sos_idx, self.eos_idx, src_outputs, len_src,\
                self.config.max_length)
        return preds

