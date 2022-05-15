# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:13:47 2022

@author: Swen
"""
import numpy as np

from .utils import stack_plot

class opt_res:
    def __init__(self, x, method):
        self.x_opt = x
        self.x_mean = np.mean(x, axis = 2).T
        self.method = method

    def __repr__(self):
        self.plot()
        return 'Optimal solution using: ' + self.method
        #return str(self.x_mean)

    def plot(self):
        stack_plot(self.x_opt, self.method)
