# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:45:40 2022

@author: Swen
"""
from .annuity import annuity
import numpy as np

def create_W_grid(kappa, rf, T, J = 11, d = 0.15, W_init = 0):
    #Alternatively:
    # J = 11
    #contribution = annuity(rf - 1,  np.linspace(1, T, T), kappa)
    #minmax = np.exp(np.linspace(-3, 3, J))
    #W_grid = np.outer(np.concatenate([np.zeros(1), contribution]), minmax)

    # #Construct W grid
    minmax = np.linspace(1 - d, 1 + d, J) * rf - 1
    W_grid = np.zeros((T, J))
    for j in range(J):
        W_grid[:, j] = annuity(minmax[j] , np.linspace(0, T - 1, T), kappa, W_init)

    return W_grid
