# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:20:15 2022

@author: Swen
"""
import numpy as np
import matplotlib.pyplot as plt

def stack_plot(x_opt, title = None):
    #x_opt = x
    T = x_opt.shape[1]
    x_ax = list(range(T))
    y = np.mean(x_opt, axis = 2).T
    y_stack = np.cumsum(y, axis = 1)
    y_stack = np.concatenate((np.zeros(T).reshape(T, 1), y_stack), axis =1)
    y_stack = np.concatenate((y_stack, np.ones(T).reshape(T, 1)), axis = 1)
    y_stack = y_stack.T

    K = len(y_stack) - 1
    if K == 2:
        names = ['Stock', 'Cash']
    elif K == 3:
        names = ['Stock', 'Bonds', 'Cash']
    else:
        raise ValueError('Number of dimensions not supported for legend')

    plt.figure()
    for i in range(K):
        plt.fill_between(x_ax, y_stack[i], y_stack[i + 1])
    plt.title(title)
    plt.xlabel("Horizon")
    plt.ylabel("Asset weight")
    plt.legend(names, loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.show()
