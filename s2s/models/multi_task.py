#!/usr/bin/env python
# coding=utf-8


import os
import time


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from s2s.models.common import Embedding
from s2s.models.encoder import RNNEncoder
from s2s.models.decoder import RNNDecoder, AttnRNNDecoder
 
class EnLMSeq2seq(nn.Module):
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

        self.encoder = RNNEncoder(
                cf.hidden_size,
                cf.num_layers,
                encoder_embed,
                cf.bidirectional,
                cf.rnn_dropout,)

        self.decoder = AttnRNNDecoder(
                cf.hidden_size,
                cf.num_layers,
                decoder_embed,
                cf.mlp1_size,
                cf.rnn_dropout,
                cf.mlp_dropout)

        self.num_directions = 2 if cf.bidirectional else 1
        self.encoder_linear = nn.Linear(cf.hidden_size, len(src_vocab))
        params = list(self.encoder.parameters()) +  list(self.decoder.parameters())
        self.optimizer = getattr(torch.optim, cf.optimizer)(params, **cf.optimizer_kwargs)
        
        self.loss = nn.CrossEntropyLoss(reduction='elementwise_mean', ignore_index=dc.pad_idx)

    def forward(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, tgt_out, len_tgt = batch['tgt_in'], batch['tgt_out'], batch['len_tgt']
        src_last, src_output = self.encoder(src, len_src)
        logits = self.decoder(tgt_in, len_tgt, src_last, src_output, len_src)
        encoder_logits = self.encoder_linear(src_output)
        return logits, encoder_logits

    def train_step(self, batch):
        logits, encoder_logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        en_seq_len = encoder_logits.size(1)
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        lm_loss = self.loss(input=encoder_logits.view(batch_size*en_seq_len, -1),\
                target=batch['src_out'][:,:en_seq_len].contiguous().view(-1))
        self.optimizer.zero_grad()
        total_loss = loss + self.config.lm_coef * lm_loss
        total_loss.backward()
        self.optimizer.step()
        return loss.item(), batch_size

    def get_loss(self, batch):
        logits, _ = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        return loss.item(), batch_size  

    def greedy_decode(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, len_tgt = batch['tgt_in'],  batch['len_tgt']
        src_last, src_outputs = self.encoder(src, len_src)
        preds = self.decoder.greedy_decode(src_last, self.sos_idx, self.eos_idx, src_outputs, len_src)
        return preds

      
 
class BiClfSeq2seq(nn.Module):
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


        self.encoder = RNNEncoder(
                cf.hidden_size,
                cf.num_layers,
                encoder_embed,
                cf.bidirectional,
                cf.rnn_dropout)

        self.clf_encoder = RNNEncoder(
                cf.hidden_size,
                cf.num_layers,
                encoder_embed,
                cf.bidirectional,
                cf.rnn_dropout)
        

        self.decoder = AttnRNNDecoder(
                cf.hidden_size,
                cf.num_layers,
                decoder_embed,
                cf.mlp1_size,
                cf.rnn_dropout,
                cf.mlp_dropout)
        
        self.num_directions = 2 if cf.bidirectional else 1
        self.encoder_linear = nn.Linear(cf.hidden_size, 3)
        self.params = list(self.encoder.parameters()) +  list(self.decoder.parameters())\
                +list(self.clf_encoder.parameters())
        self.optimizer = getattr(torch.optim, cf.optimizer)(self.params, **cf.optimizer_kwargs)
        
        self.loss = nn.CrossEntropyLoss(reduction='elementwise_mean', ignore_index=dc.pad_idx)

    def forward(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, tgt_out, len_tgt = batch['tgt_in'], batch['tgt_out'], batch['len_tgt']
        src_last, src_output = self.encoder(src, len_src)
        logits = self.decoder(tgt_in, len_tgt, src_last, src_output, len_src)
        clf_last, clf_output = self.clf_encoder(src, len_src)
        encoder_logits = self.encoder_linear(clf_output)
        return logits, encoder_logits

    def train_step(self, batch):
        logits, encoder_logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        en_seq_len = encoder_logits.size(1)
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        clf_loss = self.loss(input=encoder_logits.view(batch_size*en_seq_len, -1),\
                target=batch['src_out'][:,:en_seq_len].contiguous().view(-1))
        self.optimizer.zero_grad()
        total_loss = loss + self.config.clf_coef *clf_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.config.clip_norm)
        self.optimizer.step()
        return loss.item(), self.config.clf_coef*clf_loss.item(), batch_size

    def get_loss(self, batch):
        #logits, _ = self.forward(batch)
        logits, encoder_logits = self.forward(batch)
        en_seq_len = encoder_logits.size(1)
        batch_size, seq_len, _ = logits.size()
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        clf_loss = self.loss(input=encoder_logits.view(batch_size*en_seq_len, -1),\
                target=batch['src_out'][:,:en_seq_len].contiguous().view(-1))

        return loss.item(), self.config.clf_coef*clf_loss.item(), batch_size  

    def greedy_decode(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, len_tgt = batch['tgt_in'],  batch['len_tgt']
        src_last, src_outputs = self.encoder(src, len_src)
        preds = self.decoder.greedy_decode(src_last, self.sos_idx, self.eos_idx, src_outputs, len_src)
        return preds

  
class BiClfSeq2seqExt(nn.Module):
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


        self.encoder = RNNEncoder(
                cf.hidden_size,
                cf.num_layers,
                encoder_embed,
                cf.bidirectional,
                cf.rnn_dropout)

        self.clf_encoder = RNNEncoder(
                cf.hidden_size,
                cf.num_layers,
                encoder_embed,
                cf.bidirectional,
                cf.rnn_dropout)
        

        self.decoder = AttnRNNDecoder(
                cf.hidden_size,
                cf.num_layers,
                decoder_embed,
                #cf.mlp1_size,
                cf.rnn_dropout,
                cf.mlp_dropout)
        
        self.num_directions = 2 if cf.bidirectional else 1
        self.encoder_linear = nn.Linear(cf.hidden_size, 3)
        self.output_linear = nn.Linear(cf.hidden_size*2, cf.hidden_size)
        self.params = list(self.encoder.parameters()) +  list(self.decoder.parameters())\
                +list(self.clf_encoder.parameters())
        self.optimizer = getattr(torch.optim, cf.optimizer)(self.params, **cf.optimizer_kwargs)
        
        self.loss = nn.CrossEntropyLoss(reduction='elementwise_mean', ignore_index=dc.pad_idx)

    def forward(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, tgt_out, len_tgt = batch['tgt_in'], batch['tgt_out'], batch['len_tgt']
        src_last, src_output = self.encoder(src, len_src)
        clf_last, clf_output = self.clf_encoder(src, len_src)
        encoder_logits = self.encoder_linear(clf_output)
        cat_output = torch.tanh(self.output_linear(torch.cat((src_output, clf_output), -1)))
        logits = self.decoder(tgt_in, len_tgt, src_last, cat_output, len_src)
        return logits, encoder_logits

    def train_step(self, batch):
        logits, encoder_logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        en_seq_len = encoder_logits.size(1)
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        clf_loss = self.loss(input=encoder_logits.view(batch_size*en_seq_len, -1),\
                target=batch['src_out'][:,:en_seq_len].contiguous().view(-1))
        total_loss = loss + self.config.clf_coef *clf_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.config.clip_norm)
        self.optimizer.step()
        return loss.item(), self.config.clf_coef*clf_loss.item(), batch_size

    def get_loss(self, batch):
        #logits, _ = self.forward(batch)
        logits, encoder_logits = self.forward(batch)
        en_seq_len = encoder_logits.size(1)
        batch_size, seq_len, _ = logits.size()
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        clf_loss = self.loss(input=encoder_logits.view(batch_size*en_seq_len, -1),\
                target=batch['src_out'][:,:en_seq_len].contiguous().view(-1))

        return loss.item(), self.config.clf_coef*clf_loss.item(), batch_size  

    def greedy_decode(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, len_tgt = batch['tgt_in'],  batch['len_tgt']
        src_last, src_output = self.encoder(src, len_src)
        clf_last, clf_output = self.clf_encoder(src, len_src)
        encoder_logits = self.encoder_linear(clf_output)
        cat_output = troch.tanh(self.output_linear(torch.cat((src_output, clf_output), -1)))
        preds = self.decoder.greedy_decode(src_last, self.sos_idx, self.eos_idx, cat_output, len_src)
        return preds

          
