#!/usr/bin/env python
# coding=utf-8



import torch
import numpy as np

class BatchIterator:
    def __init__(self, generator, data_size):
       self.iterator = generator()
        
