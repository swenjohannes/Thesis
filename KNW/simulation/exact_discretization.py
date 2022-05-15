# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:01:06 2022

@author: Swen
"""

import numpy as np
from numba import njit
from .sim_res import sim_res


@njit(cache = True)
def alpha(X):
    return [(np.exp(x) - 1) / x if x != 0 else 1 for x in X]

alpha(np.ones(3)) #Run once to compile

def exact_discretization(knw_pars, T = 40, M = 10000, h = 1):
    #knw_pars = knw.pars
    #knw_pars = pars_0

    #T = 1000
    theta0, theta1, Sigma_y, BN_5 = knw_pars.set_ou_pars() #Get theta0, theta1 etc

    #import scipy
    #M = scipy.linalg.eig(theta1)
    Y = np.zeros((6, T + 1, M))
    d, U = np.linalg.eig(theta1)  # a = U @ np.diag(d) @ np.linalg.inv(U) #CHECK!
    U_inv = np.linalg.inv(U)


    #scipy.linalg.expm(np.diag(d))

    ###### ALTERNATIVE 1: determine h one time, simulate each step

    #np.identity(6) + np.diag(d) ** 1 + 1/2 * np.diag(d) ** 2 + 1/6 * np.diag(d) ** 3
    #expm(theta1)
    gamma_h = U @ np.diag(np.exp(d * h)) @ U_inv

    F = h * np.diag( alpha(d * h))
    mu_h = U @ F @ U_inv @ theta0
    mu_h = mu_h.reshape(6, 1)

    inner = np.add.outer(d, d) * h
    V = U_inv @ Sigma_y @ Sigma_y.T @ U_inv.T * h * [alpha(v) for v in inner]
    sigma_h = U @ V @ U.T

    eps_h = np.random.multivariate_normal(np.zeros(6), sigma_h, (T, M))

    for h in range(T):
        Y[:, h  + 1] = mu_h.reshape(6, 1) + gamma_h @ Y[:, h] + eps_h[h].T
    return Y


### TEST SECTION
# T = 40
# M = 10000
# Y = simulate_ED(pars_0, T, M)
