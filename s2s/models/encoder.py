#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Yifan WANG <yifanwang1993@gmail.com>
#
# Distributed under terms of the MIT license.

import os
import time


import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F

from s2s.models.common import Feedforward, Embedding, sequence_mask


"""
Encoder
"""

class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, inputs, lengths, hidden=None):
        raise NotImplementedError()

    def init_weights(self):
        raise NotImplementedError()

    def load_embedding(self, embed_file):
        # TODO
        pass



class RNNEncoder(Encoder):

    def __init__(self, 
            hidden_size,
            num_layers,
            embed,
            bidirectional=True,
            dropout=0.2,
            cell='gru'):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed.embed_size
        self.vocab_size = embed.vocab_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = dropout
        self.cell = cell
        # embedding
        self.embedding = embed


        # TODO
        self.rnn = nn.GRU(
                input_size = self.embed_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional)

        self.init_weights()

    def init_weights(self):
        # RNN
        for i in range(self.num_layers):
            if self.cell == 'gru':
                K = 3
            elif self.cell == 'lstm':
                K = 4
            for k in range(K):
                s = k*self.hidden_size
                e = (k+1)*self.hidden_size
                w = getattr(self.rnn, 'weight_ih_l{0}'.format(i))[s:e]
                nn.init.orthogonal_(w)
                w = getattr(self.rnn, 'weight_hh_l{0}'.format(i))[s:e]
                nn.init.orthogonal_(w)
                if self.bidirectional:
                    w = getattr(self.rnn, 'weight_ih_l{0}_reverse'.format(i))[s:e]
                    nn.init.orthogonal_(w)
                    w = getattr(self.rnn, 'weight_hh_l{0}_reverse'.format(i))[s:e]
                    nn.init.orthogonal_(w)


    def forward(self, inputs, lengths, hidden=None, concat_output=True):
        """
        Return:
            last (batch_size, hidden_size)
            output (batch_size, seq_len, hidden_size)
        """
        concat_output = False
        inputs = self.embedding(inputs)
        lens, indices = torch.sort(lengths, 0, True)
        inputs = inputs[indices]
        outputs, h_n = self.rnn(pack(inputs, lens, 
                batch_first=True), hidden)
        outputs = unpack(outputs, batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices]
        if not concat_output:
            h_n = h_n.view(self.num_layers, self.num_directions, inputs.size(0), self.hidden_size)
            h_n = torch.mean(h_n, 1)
            outputs = torch.mean(outputs.view(outputs.size(0), outputs.size(1), self.num_directions, self.hidden_size), 2)
        return h_n, outputs


class SelfFusionEncoder(Encoder):

    def __init__(self, 
            hidden_size,
            num_layers,
            embed,
            bidirectional=True,
            dropout=0.2,
            cell='gru'):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed.embed_size
        self.vocab_size = embed.vocab_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = dropout
        self.cell = cell
        # embedding
        self.embedding = embed


        # TODO
        self.rnn = nn.GRU(
                input_size = self.embed_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional)
        self.attn_type = 'symmetric'
        if self.attn_type == 'symmetric':
            self.inter_dim = 200
            self.D = nn.Linear(hidden_size, self.inter_dim, bias=False)
            self.U = nn.Linear(self.inter_dim, self.inter_dim, bias=False)
            self.attn_out = nn.Linear(hidden_size*2, hidden_size)

        self.init_weights()


    def init_weights(self):
        def init_rnn(rnn):
            # RNN
            for i in range(self.num_layers):
                if self.cell == 'gru':
                    K = 3
                elif self.cell == 'lstm':
                    K = 4
                for k in range(K):
                    s = k*self.hidden_size
                    e = (k+1)*self.hidden_size
                    w = getattr(rnn, 'weight_ih_l{0}'.format(i))[s:e]
                    nn.init.orthogonal_(w)
                    w = getattr(rnn, 'weight_hh_l{0}'.format(i))[s:e]
                    nn.init.orthogonal_(w)
                    if self.bidirectional:
                        w = getattr(rnn, 'weight_ih_l{0}_reverse'.format(i))[s:e]
                        nn.init.orthogonal_(w)
                        w = getattr(rnn, 'weight_hh_l{0}_reverse'.format(i))[s:e]
                        nn.init.orthogonal_(w)
            return rnn
        self.rnn = init_rnn(self.rnn)


    def forward(self, inputs, lengths, hidden=None, concat_output=True):
        """
        Return:
            last (batch_size, hidden_size)
            output (batch_size, seq_len, hidden_size)
        """
        concat_output = False
        inputs = self.embedding(inputs)
        lens, indices = torch.sort(lengths, 0, True)
        inputs = inputs[indices]
        outputs, h_n = self.rnn(pack(inputs, lens, 
                batch_first=True), hidden)
        outputs = unpack(outputs, batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices]
        if not concat_output:
            h_n = h_n.view(self.num_layers, self.num_directions, inputs.size(0), self.hidden_size)
            h_n = torch.mean(h_n, 1)
            outputs = torch.mean(outputs.view(outputs.size(0), outputs.size(1), self.num_directions, self.hidden_size), 2)
        
        mask = sequence_mask(lengths, outputs.size(1)).unsqueeze(1)
        if self.attn_type == 'symmetric':
            outputs_ = F.relu(self.D(outputs))
            scores = torch.matmul(outputs_, outputs_.transpose(1,2)) # batch_size, tgt_len, src_len
            scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, -1)
            weighted = torch.matmul(p_attn, outputs)
            new_outputs = F.tanh(self.attn_out(torch.cat([outputs, outputs*weighted], -1)))

        
        return h_n, new_outputs


class CNNEncoder(Encoder):

    def __init__(self, config):
        pass
    
    def forward(self, inputs, lengths, hidden=None):
        pass


class TransformerEncoder(Encoder):

    def __init__(self, config):
        pass

    def forward(self, inputs, lengths, hidden=None):
        pass





