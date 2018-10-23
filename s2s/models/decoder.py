#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright Â© 2018 Yifan WANG <yifanwang1993@gmail.com>
#
# Distributed under terms of the MIT license.

import os
import time


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from s2s.models.common import Feedforward, Embedding


"""
Encoder
"""

class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()


    def step(self, inputs, lengths, hidden=None, context=None, context_lengths=None, tf_ratio=1.0):
        """ One-step computation
        
        Args:
            inputs: batch x 1 x dim

        Returns:
            output: hidden state of last layer
            hidden: hidden states of all layers
        """
        pass

    def forward(self, inputs, lengths, hidden=None, context=None, context_lengths=None, tf_ratio=1.0):
        raise NotImplementedError()

    def init_weights(self):
        pass

    def greedy_decode(self, hidden, sos_idx, eos_idx, context=None, context_lengths=None, max_length=10):
        """
        inputs (batch, 1, dim)
        """
        batch_size = hidden.size(1)
        logits = []
        preds = []
        inp = sos_idx * torch.ones((batch_size, 1), dtype=torch.long)
        while True:
            logit, output, hidden = self.step(inp, hidden)
            logp = F.log_softmax(logit, dim=2)
            maxv, maxidx = torch.max(logp, dim=2)
            if ((maxidx == eos_idx).all() and len(preds) > 0)  or len(preds) >= max_length:
                break
            logits.append(logit)
            preds.append(maxidx.numpy())
            inp = maxidx
        logits = torch.cat(logits, dim=1)
        preds = np.concatenate(preds, 1)
        # print(preds.shape)
        return preds
        
    
    def beam_search(self, inputs, lengths, hidden=None, context=None, context_lengths=None):
        pass

    def sample(self, inputs, lengths, hidden=None, context=None, context_lengths=None, temperature=None):
        pass
        

class RNNDecoder(Decoder):

    def __init__(self,
            hidden_size,
            num_layers,
            embed_size,
            vocab,
            dropout=0.0,
            pretrained=None,
            pretrained_size=None,
            projection=False):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.pad_idx = vocab.pad_idx
        self.dropout = dropout
        self.pretrained = pretrained
        self.pretrained_size = pretrained_size
        self.projection = projection

        self.embedding = Embedding(
                self.embed_size,
                self.vocab,
                self.pretrained,
                self.pretrained_size,
                self.projection)

        # TODO
        self.rnn = nn.GRU(
                input_size = self.embed_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first=True,
                dropout=dropout)

        # TODO: flexible hidden_size
        self.linear_out = nn.Linear(
                self.hidden_size,
                self.vocab_size)

    def step(self, inputs, hidden=None, context=None, context_lengths=None):
        inputs = self.embedding(inputs)
        batch_size, _, _ = inputs.size()
        # output: (batch, 1, hidden_size)
        output, h_n = self.rnn(inputs, hidden)
        # (batch, 1, vocab_size)
        logit = self.linear_out(output.squeeze(1))
        return logit.unsqueeze(1), output, h_n

    def forward(self, inputs, lengths, hidden, context=None, context_lengths=None, tf_ratio=1.0):
        """
        
        """
        batch_size, seq_len = inputs.size()
        embedded = self.embedding(inputs)
        batch_size, seq_len, _ = embedded.size()
        outputs, h_n = self.rnn(embedded, hidden)

        logits = self.linear_out(outputs)
        inputs = inputs.split(1, 1)
        logits = []
        outputs = []
        for inp in inputs:
            logit, output, hidden = self.step(inp, hidden)
            logits.append(logit)
            outputs.append(output)
        outputs = torch.cat(outputs, 1)
        logits = self.linear_out(outputs)
        return logits

