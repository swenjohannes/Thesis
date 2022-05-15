# -*- coding: utf-8 -*-
"""
Created on Fri May 13 18:02:33 2022

@author: Swen
"""

from numba import njit, prange
import numpy as np

@njit
def H_inv(u, gamma):
    return 1/ (1 - gamma) * np.log((1 - gamma) * u)
#Run once to compile
H_inv(np.ones((2, 3)), 5)

@njit
def H(u, gamma):
    return np.exp(u *  (1 - gamma))/ (1 - gamma)
#Run once to compile
H(np.ones((2, 3)), 5)

@njit(parallel=True)
def smearing_est(Y, eps, gamma):
    #Y = y_H_fit; eps = eps_h;
    M = len(Y)
    est = np.zeros(M)
    for m in prange(M):
        #m = 1
        est[m] = np.mean(H(Y[m] + eps, gamma))
    return est

#Run once to compile
smearing_est(np.ones((2, 3)), np.ones((2, 3)), 5)


#H_inv(H(y, 15), 15)
