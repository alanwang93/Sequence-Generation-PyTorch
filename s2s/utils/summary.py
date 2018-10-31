#!/usr/bin/env python
# coding=utf-8

import os

class Summary:

    def __init__(self, 
            path, 
            prefix, 
            restart=False, 
            xlabel='step',
            columns=None,
            sep='\t'):

        self.path = path
        self.prefix = prefix
        self.restart = restart
        self.xlabel = xlabel
        self.columns = columns
        self.sep = sep
        
        self.file = os.path.join(path, '{0}_summary.tsv').format(self.prefix)


    def join(self, l):
        return (self.sep).join(map(str, l)) + '\n'

    
    def write(self, xval, ys):
        """
        ys: dict
        """
        if self.restart:
            if self.columns is None:
                self.columns = list(ys.keys())
            with open(self.file, 'w') as f:
                f.write(self.join([self.xlabel] + self.columns))
            self.restart = False
        else:
            with open(self.file, 'r') as f:
                self.columns = f.readline().strip().split(self.sep)
                self.xlabel = self.columns[0]
                self.columns = self.columns[1:]
        with open(self.file, 'a') as f:
            yvals = [ys[col] for col in self.columns]
            f.write(self.join([xval] + yvals))



    

