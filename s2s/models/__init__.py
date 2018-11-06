#!/usr/bin/env python
# coding=utf-8

from s2s.models.seq2seq import Seq2seq, SelfFusionSeq2seq
from s2s.models.multi_task import EnLMSeq2seq, BiClfSeq2seq, BiClfSeq2seqExt

__all__ = ['Seq2seq', 'EnLMSeq2seq', 'BiClfSeq2seq', 'BiClfSeq2seqExt', \
        'SelfFusionSeq2seq']
