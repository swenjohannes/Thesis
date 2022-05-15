# -*- coding: utf-8 -*-
"""
Created on Sun May  1 22:04:36 2022

@author: Swen
"""
import numpy as np
from numba import njit

from .mad import mad
from .ols import ols
from .wls import wls
from .Tukey_weight import Tukey_weight
from .est_res import est_res

def bisquare(Y, X, maxit = 50, eps = 1e-3):
    #k = 1.547; maxit = 1000; eps = 1e-4;
    res = ols(Y, X)
    beta_ols = res.beta

    beta_old = beta_ols
    it = 0
    try:
        while it < maxit:
            sigma_hat = 1.4826 * mad(res.e)
            k = 4.685 * sigma_hat
            u = res.e / sigma_hat

            w = Tukey_weight(u, k)      #determine weights
            res = wls(Y, X, w)  #estimate WLS
            #print(res)
            if np.linalg.norm(res.beta - beta_old) < eps :
                break
            beta_old = res.beta
            it = it + 1
    except:
        pass
        #print("Bisquare failed, using OLS!")
    if it == maxit:
        print('Bisquare failed to converge, returning OLS')
        res = est_res(Y, X, beta_ols, 'ols')
    else:
        res = est_res(Y, X, beta_old, 'bisquare') #store in est_res obj
    return res


#res_bs = bisquare(Y, X)
#res_bs.plot()
#res_ols = ols(Y, X)
#res_ols.plot()
#bisquare(Y, X)

# @timer
# def func():
#     for _ in range(10):
#         bisquare(Y, X)

# func()
