#!/usr/bin/env python
# coding=utf-8


import os
import time


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from s2s.models.common import Embedding, DecInit
from s2s.models.encoder import RNNEncoder, SelfFusionEncoder
from s2s.models.decoder import RNNDecoder, AttnRNNDecoder

class Seq2seq(nn.Module):
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
                cf.projection)
        
        decoder_embed = Embedding(
                cf.embed_size,
                tgt_vocab,
                pretrained,
                cf.pretrained_size,
                cf.projection)


        self.encoder = RNNEncoder(
                cf.hidden_size,
                cf.num_layers,
                encoder_embed,
                cf.bidirectional,
                cf.rnn_dropout)

        self.decoder = AttnRNNDecoder(
                cf.hidden_size,
                cf.num_layers,
                decoder_embed,
                cf.rnn_dropout,
                cf.mlp_dropout,
                cf.attn_type)

        self.dec_initer = DecInit(cf.hidden_size, cf.hidden_size)

        #self.num_directions = 2 if cf.bidirectional else 1
        print('params')
        for name, param in self.named_parameters():
            print(name, param.size())

        self.params = list(self.parameters()) #list(self.encoder.parameters()) +  list(self.decoder.parameters())
        self.optimizer = getattr(torch.optim, cf.optimizer)(self.params, **cf.optimizer_kwargs)
        
        self.loss = nn.CrossEntropyLoss(reduction='elementwise_mean', ignore_index=dc.pad_idx)

    def forward(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, tgt_out, len_tgt = batch['tgt_in'], batch['tgt_out'], batch['len_tgt']
        src_output, src_last = self.encoder(src, len_src)
        dec_init = self.dec_initer(src_last[1]).unsqueeze(0) # last back hidden
        logits = self.decoder(tgt_in, len_tgt, dec_init, src_output, len_src)
        return logits

    def train_step(self, batch):
        logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.config.clip_norm)
        self.optimizer.step()
        return loss.item(), batch_size

    def get_loss(self, batch):
        logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        return loss.item(), batch_size  

    def greedy_decode(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, len_tgt = batch['tgt_in'],  batch['len_tgt']
        src_outputs, src_last = self.encoder(src, len_src)
        dec_init = self.dec_initer(src_last[1]).unsqueeze(0) # last back hidden
        preds = self.decoder.greedy_decode(dec_init, self.sos_idx, self.eos_idx, src_outputs, len_src)
        return preds

 
class SelfFusionSeq2seq(nn.Module):
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
                cf.projection)
        
        decoder_embed = Embedding(
                cf.embed_size,
                tgt_vocab,
                pretrained,
                cf.pretrained_size,
                cf.projection)


        self.encoder = SelfFusionEncoder(
                cf.hidden_size,
                cf.num_layers,
                encoder_embed,
                cf.bidirectional,
                cf.rnn_dropout)

        self.decoder = AttnRNNDecoder(
                cf.hidden_size,
                cf.num_layers,
                decoder_embed,
                cf.rnn_dropout,
                cf.mlp_dropout,
                cf.attn_type)

        self.num_directions = 2 if cf.bidirectional else 1

        self.params = list(self.encoder.parameters()) +  list(self.decoder.parameters())
        self.optimizer = getattr(torch.optim, cf.optimizer)(self.params, **cf.optimizer_kwargs)
        
        self.loss = nn.CrossEntropyLoss(reduction='elementwise_mean', ignore_index=dc.pad_idx)

    def forward(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, tgt_out, len_tgt = batch['tgt_in'], batch['tgt_out'], batch['len_tgt']
        src_output, src_last = self.encoder(src, len_src)
        logits = self.decoder(tgt_in, len_tgt, src_last, src_output, len_src)
        return logits

    def train_step(self, batch):
        logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.config.clip_norm)
        self.optimizer.step()
        return loss.item(), batch_size

    def get_loss(self, batch):
        logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        return loss.item(), batch_size  

    def greedy_decode(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, len_tgt = batch['tgt_in'],  batch['len_tgt']
        src_last, src_outputs = self.encoder(src, len_src)
        preds = self.decoder.greedy_decode(src_last, self.sos_idx, self.eos_idx, src_outputs, len_src)
        return preds

 
class FusionSeq2seq(nn.Module):
    """
    At each decode step, the hidden state should contain all the rest information
    """
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
                cf.projection)
        
        decoder_embed = Embedding(
                cf.embed_size,
                tgt_vocab,
                pretrained,
                cf.pretrained_size,
                cf.projection)


        self.encoder = SelfFusionEncoder(
                cf.hidden_size,
                cf.num_layers,
                encoder_embed,
                cf.bidirectional,
                cf.rnn_dropout)

        self.decoder = AttnRNNDecoder(
                cf.hidden_size,
                cf.num_layers,
                decoder_embed,
                cf.rnn_dropout,
                cf.mlp_dropout,
                cf.attn_type)

        self.num_directions = 2 if cf.bidirectional else 1

        self.params = list(self.encoder.parameters()) +  list(self.decoder.parameters())
        self.optimizer = getattr(torch.optim, cf.optimizer)(self.params, **cf.optimizer_kwargs)
        
        self.loss = nn.CrossEntropyLoss(reduction='elementwise_mean', ignore_index=dc.pad_idx)

    def forward(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, tgt_out, len_tgt = batch['tgt_in'], batch['tgt_out'], batch['len_tgt']
        src_output, src_last = self.encoder(src, len_src)
        logits = self.decoder(tgt_in, len_tgt, src_last, src_output, len_src)
        return logits

    def train_step(self, batch):
        logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.config.clip_norm)
        self.optimizer.step()
        return loss.item(), batch_size

    def get_loss(self, batch):
        logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        return loss.item(), batch_size  

    def greedy_decode(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, len_tgt = batch['tgt_in'],  batch['len_tgt']
        src_last, src_outputs = self.encoder(src, len_src)
        preds = self.decoder.greedy_decode(src_last, self.sos_idx, self.eos_idx, src_outputs, len_src)
        return preds


