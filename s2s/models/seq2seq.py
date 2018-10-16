#!/usr/bin/env python
# coding=utf-8


import os
import time


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from s2s.models.common import BaseModel
from s2s.models.encoder import RNNEncoder
from s2s.models.decoder import RNNDecoder

class Seq2seq(nn.Module):
    def __init__(self, config):
        super().__init__()
        cf = config
        dc = cf.dataset
        
        self.config = config

        self.encoder = RNNEncoder(
                cf.hidden_size,
                cf.num_layers,
                cf.embed_size,
                cf.src_vocab_size,
                cf.bidirectional,
                pad_idx=0,
                dropout=cf.rnn_dropout,
                embed_file=cf.embed_file)

        self.decoder = RNNDecoder(
                cf.hidden_size,
                cf.num_layers,
                cf.embed_size,
                cf.tgt_vocab_size,
                cf.bidirectional,
                pad_idx=0,
                dropout=cf.rnn_dropout,
                embed_file=cf.embed_file)

        params = list(self.encoder.parameters()) +  list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params)
        
        self.loss = nn.CrossEntropyLoss(reduction='elementwise_mean', ignore_index=dc.pad_idx)

    def forward(self, batch):
        src, len_src = batch['src'], batch['len_src']
        tgt_in, tgt_out, len_tgt = batch['tgt_in'], batch['tgt_out'], batch['len_tgt']
        src_last, src_output = self.encoder(src, len_src)
        logits = self.decoder(tgt_in, len_tgt)
        return logits

    def train_step(self, batch):
        logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    
