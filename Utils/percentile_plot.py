# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:42:53 2022

@author: Swen
"""
import numpy as np
import matplotlib.pyplot as plt

def percentile_plot(y, x = None, axes = 1, title = None, xlab = None, ylab = None, marker = None, c1 = 5, c2 = 95):
    #c1 = 5; c2 = 95; ax = 1; marker = None; x = tau
    lower = np.percentile(y, c1, axis = axes)
    mean = np.mean(y, axis = axes)
    upper = np.percentile(y, c2, axis = axes)

    plt.figure()
    if x is None:
        plt.plot(upper, marker = marker, linestyle = '--')
        plt.plot(mean, marker = marker)
        plt.plot(lower, marker = marker, linestyle = '--')
    else:
        plt.plot(x, upper, marker = marker, linestyle = '--')
        plt.plot(x, mean, marker = marker)
        plt.plot(x, lower, marker = marker, linestyle = '--')
    plt.grid(True, linestyle = '--')
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(['Upper ' + str(c2) + '%',
                'Mean',
                'Lower ' + str(c1) + '%'], loc = 'upper right', prop={'size': 6.5})
