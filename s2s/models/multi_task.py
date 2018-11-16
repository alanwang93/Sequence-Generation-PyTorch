#!/usr/bin/env python
# coding=utf-8


import os
import time


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from s2s.models.common import Embedding, DecInit
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

      
 
class MTSeq2seq(nn.Module):
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
                cf.enc_num_layers,
                encoder_embed,
                cf.bidirectional,
                cf.rnn_dropout)

        #self.clf_encoder = RNNEncoder(
        #        cf.hidden_size,
        #        cf.enc_num_layers,
        #        encoder_embed,
        #        cf.bidirectional,
        #        cf.rnn_dropout)

        self.decoder = AttnRNNDecoder(
                cf.hidden_size,
                cf.dec_num_layers,
                decoder_embed,
                cf.rnn_dropout,
                cf.mlp_dropout,
                cf.attn_type)
        
        self.num_directions = 2 if cf.bidirectional else 1
        self.mt_linear = nn.Linear(cf.hidden_size*2, 3) # TODO: add sentence repr

        self.dec_initer = DecInit(
                cf.hidden_size, 
                cf.hidden_size, 
                bidirectional=cf.bidirectional)


        self.params = list(self.parameters())
        self.optimizer = getattr(torch.optim, cf.optimizer)(self.params, **cf.optimizer_kwargs)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                cf.scheduler_mode,
                factor=0.5,
                patience=cf.patience,
                verbose=True)
       
        self.loss = nn.CrossEntropyLoss(reduction='elementwise_mean', ignore_index=dc.pad_idx)
        self.mt_loss = nn.CrossEntropyLoss(weight=torch.Tensor([0., 5., 1.]), reduction='elementwise_mean', ignore_index=dc.pad_idx)

    def forward(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, tgt_out, len_tgt = batch['tgt_in'], batch['tgt_out'], batch['len_tgt']
        src_output, src_last = self.encoder(src, len_src)
        dec_init = self.dec_initer(src_last)#.unsqueeze(0)
        logits = self.decoder(tgt_in, len_tgt, dec_init, src_output, len_src)
        mt_logits = self.mt_linear(src_output)
        return logits, mt_logits

    def train_step(self, batch):
        logits, mt_logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        en_seq_len = mt_logits.size(1)
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        mt_loss = self.mt_loss(input=mt_logits.view(batch_size*en_seq_len, -1),\
                target=batch['src_out'][:,:en_seq_len].contiguous().view(-1))
        self.optimizer.zero_grad()
        total_loss = loss + self.config.mt_coef *mt_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.config.clip_norm)
        self.optimizer.step()
        return loss.item(), mt_loss.item(), batch_size

    def get_loss(self, batch):
        #logits, _ = self.forward(batch)
        logits, mt_logits = self.forward(batch)
        en_seq_len = mt_logits.size(1)
        batch_size, seq_len, _ = logits.size()
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        mt_loss = self.mt_loss(input=mt_logits.view(batch_size*en_seq_len, -1),\
                target=batch['src_out'][:,:en_seq_len].contiguous().view(-1))
        #mt_preds = torch.argmax(encoder_logits.view(batch_size*en_seq_len, -1), dim=1).cpu().numpy()
        #y_true = batch['src_out'][:,:en_seq_len].contiguous().view(-1).cpu().numpy()
        #print('pred', mt_preds[:30])
        #print('true', y_true[:30], '\n')
        #n_true = 0
        #for t, p in zip(y_true, mt_preds):
        #    if t == p and t != 0:
        #        n_true += 1
        #prec = n_true/len(y_true)
        #print('precision', prec)

        return loss.item(), mt_loss.item(), batch_size  

    def greedy_decode(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, len_tgt = batch['tgt_in'],  batch['len_tgt']
        src_outputs, src_last = self.encoder(src, len_src)
        dec_init = self.dec_initer(src_last)#.unsqueeze(0)
        preds = self.decoder.greedy_decode(dec_init, self.sos_idx, self.eos_idx, src_outputs, len_src)
        return preds


  
class MTSeq2seq_v2(nn.Module):
    """
    An another RNN predicts saliency of words, only the embeddings
    are shared
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
                cf.enc_num_layers,
                encoder_embed,
                cf.bidirectional,
                cf.rnn_dropout)

        self.mt_size = cf.hidden_size//2

        self.mt_encoder = RNNEncoder(
                self.mt_size, # a smaller one
                cf.enc_num_layers,
                encoder_embed,
                cf.bidirectional, # True
                cf.rnn_dropout)
        
        self.decoder = AttnRNNDecoder(
                cf.hidden_size,
                cf.dec_num_layers,
                decoder_embed,
                cf.rnn_dropout,
                cf.mlp_dropout,
                cf.attn_type)

        self.dec_initer = DecInit(
                cf.hidden_size, 
                cf.hidden_size, 
                bidirectional=cf.bidirectional)

        
        self.mt_linear = nn.Linear(self.mt_size*2, 3)
        #self.gate = nn.Linear(cf.hidden_size*4 +self.mt_size*2 , cf.hidden_size*2)
        #self.output_linear = nn.Linear(cf.hidden_size*2, cf.hidden_size)
        self.params = list(self.parameters())
        self.optimizer = getattr(torch.optim, cf.optimizer)(self.params, **cf.optimizer_kwargs)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                cf.scheduler_mode,
                factor=0.5,
                patience=cf.patience,
                verbose=True)
       
        self.loss = nn.CrossEntropyLoss(reduction='elementwise_mean', ignore_index=dc.pad_idx)
        self.mt_loss = nn.CrossEntropyLoss(weight=torch.Tensor([0., 5., 1.]), reduction='elementwise_mean', ignore_index=dc.pad_idx)
    
    
    def forward(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, tgt_out, len_tgt = batch['tgt_in'], batch['tgt_out'], batch['len_tgt']
        src_output, src_last = self.encoder(src, len_src)
        #forward_last = src_last[0]
        #backward_last = src_last[1]
        #sentence_vector = torch.cat((forward_last, backward_last), dim=-1)
        mt_output, mt_last = self.mt_encoder(src, len_src)
        batch_size, mt_len, dim = mt_output.size()

        mt_logits = self.mt_linear(mt_output)
        # gate
        #gate = F.sigmoid(self.gate(torch.cat(
        #    (src_output, sentence_vector.unsqueeze(1).expand_as(src_output), mt_output), -1)))
        #gated = src_output * gate 
        dec_init = self.dec_initer(src_last)#.unsqueeze(0)
        #cat_output = torch.tanh(self.output_linear(torch.cat((src_output, clf_output), -1)))
        logits = self.decoder(tgt_in, len_tgt, dec_init, src_output, len_src)
        return logits, mt_logits

    def train_step(self, batch):
        logits, mt_logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        en_seq_len = mt_logits.size(1)
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        mt_loss = self.mt_loss(input=mt_logits.view(batch_size*en_seq_len, -1),\
                target=batch['src_out'][:,:en_seq_len].contiguous().view(-1))
        total_loss = loss + self.config.mt_coef *mt_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.config.clip_norm)
        self.optimizer.step()
        return loss.item(), mt_loss.item(), batch_size

    def get_loss(self, batch):
        logits, mt_logits = self.forward(batch)
        en_seq_len = mt_logits.size(1)
        batch_size, seq_len, _ = logits.size()
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        mt_loss = self.mt_loss(input=mt_logits.view(batch_size*en_seq_len, -1),\
                target=batch['src_out'][:,:en_seq_len].contiguous().view(-1))

        return loss.item(), mt_loss.item(), batch_size  

    def greedy_decode(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, len_tgt = batch['tgt_in'],  batch['len_tgt']
        src_output, src_last = self.encoder(src, len_src)
        #mt_output, mt_last = self.mt_encoder(src, len_src)
        #forward_last = src_last[0]
        #backward_last = src_last[1]
        #sentence_vector = torch.cat((forward_last, backward_last), dim=-1)

        #mt_logits = self.mt_linear(clf_output)
        #gate = F.sigmoid(self.gate(torch.cat(
        #    (src_output, sentence_vector.unsqueeze(1).expand_as(src_output), mt_output), -1)))
        #gated = src_output * gate 
        dec_init = self.dec_initer(src_last)#.unsqueeze(0)
        preds = self.decoder.greedy_decode(dec_init, self.sos_idx, self.eos_idx, src_output, len_src)
        return preds


  
class GatedMTSeq2seq(nn.Module):
    """
    An another RNN predicts saliency of words, and its hidden states will be 
    use to generate a gate together with encoder states and sentence representation 
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
                cf.enc_num_layers,
                encoder_embed,
                cf.bidirectional,
                cf.rnn_dropout)

        self.mt_size = cf.hidden_size//2

        self.mt_encoder = RNNEncoder(
                self.mt_size, # a smaller one
                cf.enc_num_layers,
                encoder_embed,
                cf.bidirectional, # True
                cf.rnn_dropout)
        
        self.decoder = AttnRNNDecoder(
                cf.hidden_size,
                cf.dec_num_layers,
                decoder_embed,
                cf.rnn_dropout,
                cf.mlp_dropout,
                cf.attn_type)

        self.dec_initer = DecInit(
                cf.hidden_size, 
                cf.hidden_size, 
                bidirectional=cf.bidirectional)

        
        self.mt_linear = nn.Linear(cf.hidden_size*2 + self.mt_size*2, 3)
        self.gate = nn.Linear(cf.hidden_size*4 +self.mt_size*2 , cf.hidden_size*2)
        #self.output_linear = nn.Linear(cf.hidden_size*2, cf.hidden_size)
        self.params = list(self.parameters())
        self.optimizer = getattr(torch.optim, cf.optimizer)(self.params, **cf.optimizer_kwargs)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                cf.scheduler_mode,
                factor=0.5,
                patience=cf.patience,
                verbose=True)
       
        self.loss = nn.CrossEntropyLoss(reduction='elementwise_mean', ignore_index=dc.pad_idx)
        self.mt_loss = nn.CrossEntropyLoss(weight=torch.Tensor([0., 5., 1.]), reduction='elementwise_mean', ignore_index=dc.pad_idx)
    
    
    def forward(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, tgt_out, len_tgt = batch['tgt_in'], batch['tgt_out'], batch['len_tgt']
        src_output, src_last = self.encoder(src, len_src)
        forward_last = src_last[0]
        backward_last = src_last[1]
        sentence_vector = torch.cat((forward_last, backward_last), dim=-1)
        mt_output, mt_last = self.mt_encoder(src, len_src)
        batch_size, mt_len, dim = mt_output.size()

        mt_logits = self.mt_linear(torch.cat((mt_output, sentence_vector.unsqueeze(1).expand(batch_size, mt_len, -1)), -1))
        # gate
        gate = F.sigmoid(self.gate(torch.cat(
            (src_output, sentence_vector.unsqueeze(1).expand_as(src_output), mt_output), -1)))
        gated = src_output * gate 
        dec_init = self.dec_initer(src_last)#.unsqueeze(0)
        #cat_output = torch.tanh(self.output_linear(torch.cat((src_output, clf_output), -1)))
        logits = self.decoder(tgt_in, len_tgt, dec_init, gated, len_src)
        return logits, mt_logits

    def train_step(self, batch):
        logits, mt_logits = self.forward(batch)
        batch_size, seq_len, _ = logits.size()
        en_seq_len = mt_logits.size(1)
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        mt_loss = self.mt_loss(input=mt_logits.view(batch_size*en_seq_len, -1),\
                target=batch['src_out'][:,:en_seq_len].contiguous().view(-1))
        total_loss = loss + self.config.mt_coef *mt_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.config.clip_norm)
        self.optimizer.step()
        return loss.item(), mt_loss.item(), batch_size

    def get_loss(self, batch):
        logits, mt_logits = self.forward(batch)
        en_seq_len = mt_logits.size(1)
        batch_size, seq_len, _ = logits.size()
        loss = self.loss(input=logits.view(batch_size*seq_len, -1), target=batch['tgt_out'].view(-1))
        mt_loss = self.mt_loss(input=mt_logits.view(batch_size*en_seq_len, -1),\
                target=batch['src_out'][:,:en_seq_len].contiguous().view(-1))

        return loss.item(), mt_loss.item(), batch_size  

    def greedy_decode(self, batch):
        src, len_src = batch['src_in'], batch['len_src']
        tgt_in, len_tgt = batch['tgt_in'],  batch['len_tgt']
        src_output, src_last = self.encoder(src, len_src)
        mt_output, mt_last = self.mt_encoder(src, len_src)
        forward_last = src_last[0]
        backward_last = src_last[1]
        sentence_vector = torch.cat((forward_last, backward_last), dim=-1)

        #mt_logits = self.mt_linear(clf_output)
        gate = F.sigmoid(self.gate(torch.cat(
            (src_output, sentence_vector.unsqueeze(1).expand_as(src_output), mt_output), -1)))
        gated = src_output * gate 
        dec_init = self.dec_initer(src_last)#.unsqueeze(0)
        preds = self.decoder.greedy_decode(dec_init, self.sos_idx, self.eos_idx, gated, len_src)
        return preds

          
