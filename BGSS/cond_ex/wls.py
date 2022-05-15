# -*- coding: utf-8 -*-
"""
Created on Sun May  1 22:02:17 2022

@author: Swen
"""
import numpy as np
from .est_res import est_res

def wls(Y, X, w):
    """
        Fits WLS. w is a vector that is diagonalized to obtain W. Returns an estimation result (est_res) object.
    """
    W = np.diag(w)
    beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
    res = est_res(Y, X, beta, 'wls')
    return res
