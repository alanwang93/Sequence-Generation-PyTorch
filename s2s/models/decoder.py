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

from s2s.models.common import Feedforward, Embedding, sequence_mask, Attention, StackedGRU, \
        ScoreLayer


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
        mask = sequence_mask(context_lengths, context.size(1)).unsqueeze(1)
        inp = sos_idx * torch.ones((batch_size,), dtype=torch.long).to(hidden.device)
        while True:
            embed_t = self.embedding(inp)
            logit, hidden, weighted, p_attn = self.step(embed_t, hidden, context, mask)
            logp = F.log_softmax(logit, dim=-1)
            maxv, maxidx = torch.max(logp, dim=-1)
            if ((maxidx == eos_idx).all() and len(preds) > 0)  or len(preds) >= max_length:
                break
            logits.append(logit)
            preds.append(maxidx.unsqueeze(1).cpu().numpy())
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
            embed,
            rnn_dropout=0.,
            mlp_dropout=0.,
            attn_type=None):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
      #  self.embed = embed
        self.vocab_size = embed.vocab_size
        self.embed_size = embed.embed_size

        #self.dropout = dropout
        self.embedding = embed

        self.cell = 'gru'
        # TODO
        #self.rnn = nn.GRU(
        #        input_size = self.embed_size,
        #        hidden_size = self.hidden_size,
        #        num_layers = self.num_layers,
        #        batch_first=True,
        #        dropout=rnn_dropout)

        self.rnn = StackedGRU(
                self.embed_size,
                self.hidden_size,
                self.num_layers,
                rnn_dropout)

        # TODO: flexible hidden_size
        self.mlp1 = nn.Linear(
                self.hidden_size,
                self.vocab_size)

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


    def step(self, embed_t, hidden, context=None, mask=None):
        """
        embed_t: batch x embed_size
        hidden: 
        """
        # output: (batch, 1, hidden_size)
        output, h_n = self.rnn(embed_t, hidden)
        # (batch, 1, vocab_size)
        logit = self.mlp1(output.squeeze(1))
        return logit.unsqueeze(1), output, h_n


    def forward(self, inputs, lengths, hidden, context=None, context_lengths=None, tf_ratio=1.0):
        """
        
        """
        #batch_size, seq_len = inputs.size()
        embedded = self.embedding(inputs)
        outputs, h_n = self.rnn(embedded, hidden)
        logits = self.mlp1(outputs)
        return logits



class AttnRNNDecoder(Decoder):

    def __init__(self,
            hidden_size,
            num_layers,
            embed,
            rnn_dropout=0., 
            mlp_dropout=0.,
            attn_type='bilinear'):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed.embed_size
        self.vocab_size = embed.vocab_size
        self.attn_type = attn_type

        self.embedding = embed
        self.cell = 'gru'

        self.rnn = StackedGRU(
                self.embed_size,
                self.hidden_size,
                self.num_layers,
                rnn_dropout)

        # TODO: flexible hidden_size
        self.mlp1 = nn.Linear(
                self.hidden_size,
                self.vocab_size)

        self.dropout = nn.Dropout(mlp_dropout)
        self.attn = Attention(
                hidden_size*2, # bi-GRU
                hidden_size,
                attn_type,
                hidden_size)
        self.score_layer = 'readout'
        self.score_layer_ = ScoreLayer(
                self.vocab_size,
                hidden_size*2,
                hidden_size,
                self.embed_size,
                self.score_layer)

        #self.init_weights()

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


    def step(self, embed_t, hidden, context=None, mask=None):
        """

        """
        output, h_n = self.rnn(embed_t, hidden)
        weighted, p_attn = self.attn(output, context, mask)
        logit = self.score_layer_(output, weighted, embed_t)
        return logit, h_n, weighted, p_attn

    def forward(self, inputs, lengths, hidden, context=None, context_lengths=None, tf_ratio=1.0):
        """
        
        """
        batch_size, seq_len = inputs.size()
        mask = sequence_mask(context_lengths, context.size(1)).unsqueeze(1)
        #embedded = self.embedding(inputs)
        #outputs = []
        logits = []
        embedded = self.embedding(inputs)
        for embed_t in embedded.split(1, dim=1):
            embed_t = embed_t.squeeze(1)
            logit, hidden, weighted, p_attn = self.step(embed_t, hidden, context, mask)
            logits.append(logit)
        logits = torch.stack(logits, 1).squeeze(2)
        return logits

"""
        outputs, h_n = self.rnn(embedded, hidden)

        mask = sequence_mask(context_lengths, context.size(1)).unsqueeze(1)

        # batch_size, seq_len, dim
        if self.attn_type == 'symmetric':
            outputs_ = F.relu(self.D(outputs))
            context_ = F.relu(self.D(context))
            scores = torch.matmul(outputs_, context_.transpose(1,2)) # batch_size, tgt_len, src_len
            scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, -1)
            weighted = torch.matmul(p_attn, context)
            new_outputs = F.tanh(self.attn_out(torch.cat([outputs, weighted], -1)))
        elif self.attn_type == 'add':
            scores = torch.tanh(self.Wh(context).unsqueeze(1).expand(-1, outputs.size(1),-1,-1)\
                    +  self.Ws(outputs).unsqueeze(2).expand(-1, -1, context.size(1), -1))
            scores = self.v(scores).squeeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, -1)
            weighted = torch.matmul(p_attn, context)
            new_outputs = F.tanh(self.attn_out(torch.cat([outputs, weighted], -1)))
        elif self.attn_type == 'bilinear':
            scores = self.W(context) # batch, src_len, dim
            scores = torch.matmul(outputs, scores.transpose(1,2))
            scores = scores.masked_fill(mask == 0, -1e9)
            # batch, tgt_len, src_len
            p_attn = F.softmax(scores, -1)
            weighted = torch.matmul(p_attn, context)
            
        if self.score_layer == '2linear':
            new_outputs = self.attn_out(torch.cat([outputs, weighted], -1))
            new_outputs = self.dropout(new_outputs)
            logits = self.mlp1(new_outputs)
            logits = self.dropout(logits)
"""


