#!/usr/bin/env python
# coding=utf-8


import os
import time


import numpy as np
import torch
import torch.nn as nn

from s2s.models.common import BaseModel
from s2s.models.encoder import RNNEncoder
from s2s.models.decoder import RNNDecoder

class Seq2seq(BaseModel):
    def __init__(self, config):
        cf = config
        dc = cf.dataset
        
        self.config = config

        self.encoder = RNNEncoder(
                cf.hidden_size,
                cf.num_layers,
                cf.embed_size,
                cf.vocab_size,
                cf.bidirectional,
                pad_idx=0,
                dropout=cf.dropout,
                embed_file=cf.embed_file)

        self.decoder = RNNDecoder(
                cf.hidden_size,
                cf.num_layers,
                cf.embed_size,
                cf.vocab_size,
                cf.bidirectional,
                pad_idx=0,
                dropout=cf.dropout,
                embed_file=cf.embed_file)

        params = list(self.encoder.parameters()) +  list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params)
        
        self.loss = nn.CrossEntropyLoss(size_average=True, ignore_index=dc.pad_dix)

    def forward(self):
        pass

    def train_step(self, batch):
        pass

    
