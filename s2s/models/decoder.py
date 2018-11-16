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
        ScoreLayer, AttentiveGate


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

    def greedy_decode(self, hidden, sos_idx, eos_idx, context=None, context_lengths=None, max_length=30):
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

class AttGateDecoder(Decoder):
    """
    FAIL
    Attentive gating
    # The deocder set a gate for encoder outputs
    Gate instead of attention
    """
    def __init__(self,
            hidden_size,
            num_layers,
            embed,
            rnn_dropout=0., 
            mlp_dropout=0.,
            attn_type='concat'):

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
        self.gate = AttentiveGate(
                hidden_size*2, # bi-GRU
                hidden_size,
                'concat',
                hidden_size)
        self.score_layer = 'readout'
        self.score_layer_ = ScoreLayer(
                self.vocab_size,
                hidden_size*2,
                hidden_size,
                self.embed_size,
                self.score_layer)

        #self.gate = nn.Linear(hidden_size*3, hidden_size*2)


    def step(self, embed_t, hidden, context=None, mask=None):
        """

        """
        output, h_n = self.rnn(embed_t, hidden)
        # gating
        #tmp = torch.cat((output.unsqueeze(1).expand(-1, context.size(1), -1), context), -1)
        #gate = self.gate(tmp)
        #context = context * gate
        weighted, gate = self.gate(output, context, mask)
        logit = self.score_layer_(output, weighted, embed_t)
        return logit, h_n, weighted, gate

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


class AGDecoder(Decoder):
    """
    NOT BAD
    Attentive gating
    The deocder set a gate for encoder outputs
    """
    def __init__(self,
            hidden_size,
            num_layers,
            embed,
            rnn_dropout=0., 
            mlp_dropout=0.,
            attn_type='concat'):

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

        self.gate = nn.Linear(hidden_size*3, hidden_size*2)


    def step(self, embed_t, hidden, context=None, mask=None):
        """

        """
        output, h_n = self.rnn(embed_t, hidden)
        # gating
        tmp = torch.cat((output.unsqueeze(1).expand(-1, context.size(1), -1), context), -1)
        gate = self.gate(tmp)
        context = context * gate
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


class AttnRNNDecoder(Decoder):
    """
    
    """
    def __init__(self,
            hidden_size,
            num_layers,
            embed,
            rnn_dropout=0., 
            mlp_dropout=0.,
            attn_type='concat'):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed.embed_size
        self.vocab_size = embed.vocab_size
        self.attn_type = attn_type

        self.embedding = embed
        self.cell = 'gru'

        self.input_feed = True
        self.input_size = self.embed_size
        if self.input_feed:
            self.input_size += self.hidden_size*2


        self.rnn = StackedGRU(
                self.input_size,
                self.hidden_size,
                self.num_layers,
                rnn_dropout)

        # TODO: flexible hidden_size
        #self.mlp1 = nn.Linear(
        #        self.hidden_size,
        #        self.vocab_size)

        #self.dropout = nn.Dropout(mlp_dropout)
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

    def make_init_att(self, context):
        return context.data.new(context.size(0), context.size(2)).zero_()

    def step(self, embed_t, hidden, context=None, mask=None, weighted=None):
        """

        """
        inp_t = embed_t
        if self.input_feed:
            inp_t = torch.cat((embed_t, weighted), 1)
        output, h_n = self.rnn(inp_t, hidden)
        weighted, p_attn = self.attn(output, context, mask)
        logit = self.score_layer_(output, weighted, embed_t)
        return logit, h_n, weighted, p_attn

    def forward(self, inputs, lengths, hidden, context=None, context_lengths=None, tf_ratio=1.0):
        """
        
        """
        weighted = self.make_init_att(context)
        batch_size, seq_len = inputs.size()
        mask = sequence_mask(context_lengths, context.size(1)).unsqueeze(1)
        logits = []
        embedded = self.embedding(inputs)
        for embed_t in embedded.split(1, dim=1):
            embed_t = embed_t.squeeze(1)
            logit, hidden, weighted, p_attn = self.step(embed_t, hidden, context, mask, weighted)
            logits.append(logit)
        logits = torch.stack(logits, 1).squeeze(2)
        return logits


class GatedAttnRNNDecoder(Decoder):
    """
    Another RNN encoder will encode the predictions (or ground truth during training)
    of our decoder, whose hidden states will be used to generate a gate for decoder hidden
    state before it's passed to next time step;
    The intuition is to explicitly let the decoder know what has been generated
    """

    def __init__(self,
            hidden_size,
            num_layers,
            embed,
            rnn_dropout=0., 
            mlp_dropout=0.,
            attn_type='concat'):

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
        
        self.enc = StackedGRU(
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
        self.gate = nn.Linear(hidden_size*2, hidden_size)


    def step(self, embed_t, hidden, context=None, mask=None):
        """

        """
        output, h_n = self.rnn(embed_t, hidden)
        weighted, p_attn = self.attn(output, context, mask)
        logit = self.score_layer_(output, weighted, embed_t)
        return logit, h_n, weighted, p_attn

    def forward(self, inputs, lengths, hidden, context=None, context_lengths=None, tf_ratio=1.0):
        """
        inputs: <SOS>, w1, w2...
        """
        batch_size, seq_len = inputs.size()
        mask = sequence_mask(context_lengths, context.size(1)).unsqueeze(1)
        #embedded = self.embedding(inputs)
        #outputs = []
        logits = []
        enc_h = None
        embedded = self.embedding(inputs).split(1, dim=1)
        for t, embed_t in enumerate(embedded):
            embed_t = embed_t.squeeze(1)
            logit, hidden, weighted, p_attn = self.step(embed_t, hidden, context, mask)
            logits.append(logit)
            if t != seq_len-1:
                enc_output, enc_h = self.enc(embedded[t+1].squeeze(1), enc_h)
                gate = F.sigmoid(self.gate(torch.cat((enc_h, hidden),-1)))
                hidden = hidden * gate

        logits = torch.stack(logits, 1).squeeze(2)
        return logits

    def greedy_decode(self, hidden, sos_idx, eos_idx, context=None, context_lengths=None, max_length=30):
        """
        inputs (batch, 1, dim)
        """
        batch_size = hidden.size(1)
        logits = []
        preds = []
        mask = sequence_mask(context_lengths, context.size(1)).unsqueeze(1)
        inp = sos_idx * torch.ones((batch_size,), dtype=torch.long).to(hidden.device)
        t = 0
        enc_h = None
        while True:
            embed_t = self.embedding(inp)
            if t > 0:
                enc_output, enc_h = self.enc(embed_t, enc_h)
                gate = F.sigmoid(self.gate(torch.cat((enc_h, hidden),-1)))
                hidden = hidden * gate
            logit, hidden, weighted, p_attn = self.step(embed_t, hidden, context, mask)
            logp = F.log_softmax(logit, dim=-1)
            maxv, maxidx = torch.max(logp, dim=-1)
            if ((maxidx == eos_idx).all() and len(preds) > 0)  or len(preds) >= max_length:
                break
            logits.append(logit)
            preds.append(maxidx.unsqueeze(1).cpu().numpy())
            inp = maxidx
            t += 1
        logits = torch.cat(logits, dim=1)
        preds = np.concatenate(preds, 1)
        # print(preds.shape)
        return preds



class MinusAttnRNNDecoder(Decoder):
    """
    Another RNN encoder will encode the predictions (or ground truth during training)
    of our decoder, whose hidden states will be used to generate a gate for decoder hidden
    state before it's passed to next time step
    """

    def __init__(self,
            hidden_size,
            num_layers,
            embed,
            rnn_dropout=0., 
            mlp_dropout=0.,
            attn_type='concat'):

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
        
        self.enc = StackedGRU(
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
        self.gate = nn.Linear(hidden_size*2, hidden_size)


    def step(self, embed_t, hidden, context=None, mask=None):
        """

        """
        output, h_n = self.rnn(embed_t, hidden)
        weighted, p_attn = self.attn(output, context, mask)
        logit = self.score_layer_(output, weighted, embed_t)
        return logit, h_n, weighted, p_attn

    def forward(self, inputs, lengths, hidden, context=None, context_lengths=None, tf_ratio=1.0):
        """
        inputs: <SOS>, w1, w2...
        """
        batch_size, seq_len = inputs.size()
        mask = sequence_mask(context_lengths, context.size(1)).unsqueeze(1)
        #embedded = self.embedding(inputs)
        #outputs = []
        logits = []
        enc_h = None
        embedded = self.embedding(inputs).split(1, dim=1)
        for t, embed_t in enumerate(embedded):
            embed_t = embed_t.squeeze(1)
            logit, hidden, weighted, p_attn = self.step(embed_t, hidden, context, mask)
            logits.append(logit)
            if t != seq_len-1:
                enc_output, enc_h = self.enc(embedded[t+1].squeeze(1), enc_h)
                gate = F.sigmoid(self.gate(torch.cat((enc_h, hidden),-1)))
                hidden = hidden * gate

        logits = torch.stack(logits, 1).squeeze(2)
        return logits

    def greedy_decode(self, hidden, sos_idx, eos_idx, context=None, context_lengths=None, max_length=30):
        """
        inputs (batch, 1, dim)
        """
        batch_size = hidden.size(1)
        logits = []
        preds = []
        mask = sequence_mask(context_lengths, context.size(1)).unsqueeze(1)
        inp = sos_idx * torch.ones((batch_size,), dtype=torch.long).to(hidden.device)
        t = 0
        enc_h = None
        while True:
            embed_t = self.embedding(inp)
            if t > 0:
                enc_output, enc_h = self.enc(embed_t, enc_h)
                gate = F.sigmoid(self.gate(torch.cat((enc_h, hidden),-1)))
                hidden = hidden * gate
            logit, hidden, weighted, p_attn = self.step(embed_t, hidden, context, mask)
            logp = F.log_softmax(logit, dim=-1)
            maxv, maxidx = torch.max(logp, dim=-1)
            if ((maxidx == eos_idx).all() and len(preds) > 0)  or len(preds) >= max_length:
                break
            logits.append(logit)
            preds.append(maxidx.unsqueeze(1).cpu().numpy())
            inp = maxidx
            t += 1
        logits = torch.cat(logits, dim=1)
        preds = np.concatenate(preds, 1)
        # print(preds.shape)
        return preds

