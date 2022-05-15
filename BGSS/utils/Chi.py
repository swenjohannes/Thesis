# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:30:05 2022

@author: Swen
"""
import numpy as np

def Chi(psi_hat, eps, mu_l):
    #
    E = np.exp(np.cumsum(eps + mu_l, axis = 0))
    chi = np.sum(E * psi_hat, axis = 0)
    return chi
