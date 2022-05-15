# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:31:05 2022

@author: Swen
"""
import numpy as np

def Psi(x_hat, Re, Rf):
    Ret = np.sum(x_hat *  Re, axis = 0) + Rf
    rev_rp = np.flip(Ret, axis = 0)  #Flip returns
    psi_t = np.cumprod(rev_rp, axis = 0)    #Compute cumprod from t to T
    psi_t = np.flip(psi_t, axis = 0)         #Flip back
    return psi_t
