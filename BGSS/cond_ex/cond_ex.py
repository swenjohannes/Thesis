# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:22:22 2022

@author: Swen
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, jit

from .ols import ols, fast_ols
from .bisquare import bisquare
from .transform import H, H_inv, smearing_est


def regress(Y, X, method):
    """
        A wrapper function to select between the available methods
    """
    if method == 'ols':
       y_hat = fast_ols(Y, X)
    elif method == 'bs':
       y_hat = bisquare(Y, X)
    else:
        raise ValueError('selected method is invalid, please select from: ols, bisquare')
    return y_hat


#@jit
def cond_ex(Y, X, method = 'ols', transform = False, gamma = None):
        """
            Returns the conditional expectation of the regression Y ~ X.

            Available methods: OLS, bisquare
        """

        #Y = y.copy(), X = basis
        if not transform:
            y_hat = regress(Y, X, method) #Normal regression
        else:
            Y_inv = H_inv(Y, gamma)
            Y_inv_hat = regress(Y_inv, X, method)
            eps_h = Y_inv - Y_inv_hat
            y_hat = smearing_est(Y_inv_hat, eps_h, gamma) #Smearing estimate

        return y_hat        #Return predicted values and the estimator


#cond_ex(y, basis)
