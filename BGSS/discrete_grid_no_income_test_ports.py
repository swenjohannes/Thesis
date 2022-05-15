# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:31:45 2022

@author: Swen
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from numpy.polynomial.polynomial import polyfit

from .utils import create_x_grid, polynomial_basis
from .cond_ex import *
from .opt_res import opt_res
from .cond_ex.transform import H, H_inv, smearing_est

@timer
def discrete_grid_no_income(Re, Rf, Z, gamma = 5, G = 50, basis_order = [2, 0, True, False, False], method = 'ols',
                            transform = False, use_test_port = True, test_ratio = 0.2):

    if np.ndim(Re) != 3:
        raise ValueError('The dimensions of Re are incorrect. Should be 3 dimensional: K assets x T periods x M observations')
    if np.ndim(Z) != 3:
        raise ValueError('The dimensions of Z are incorrect. Should be 3 dimensional: K assets x T periods x M observations')
    if np.ndim(Rf) != 2:
        raise ValueError('The dimensions of Rf are incorrect. Should be 2 dimensional: T periods x M observations')
    #gamma = 15; G = 100; basis_order = None; method = 'ols'; transform = True; use_test_port = True; test_ratio = 0.2

    K, T, M = Re.shape
    #G = 100 #G = 20 #  use_test_port = False

    #X Grid creation
    if use_test_port:
        x_grid, n_combs = create_x_grid(int(G * test_ratio), K)
        x_full, full_combs =create_x_grid(G, K) #Construct the full discrete grid

        x_test_basis = polynomial_basis(x_grid.T, order = [5, 0, False, False, False])
        x_full_basis = polynomial_basis(x_full.T, order = [5, 0, False, False, False])
    else:
        x_grid, n_combs = create_x_grid(G, K)
        x_full = x_grid

    psi = np.ones((T, M)) #/ (1 - gamma) #First psi = 1
    x = np.zeros((K, T - 1, M))     #Store opt values

    #Solve T -> 2
    for t in reversed(range(1, T)):
        #t = T - 1#t =10 #t = t -1
        print(t)
        basis = polynomial_basis(Z[:, t - 1], Re[:, t - 1], basis_order)

        beta = np.zeros((n_combs, basis.shape[1]))
        for g in range(n_combs):
            y =(np.dot(x_grid[g], Re[:, t]) + Rf[t - 1]) **  (1 - gamma) / (1 - gamma) * psi[t]
            beta[g] = fast_ols(y, basis, return_beta = True)

        if use_test_port:  #Regress beta on x basis of xgrid to approximate full set of beta's
            beta = fast_ols(beta, x_test_basis, x_new = x_full_basis)
            #plt.plot(beta)
        y_hat = basis @ beta.T #Conditional expectation
        x_opt = x_full[np.argmax(y_hat, axis = 1)].T

        x[:, t - 1] = x_opt
        psi[t - 1] =  (np.sum(x_opt * Re[:, t], axis = 0) + Rf[t - 1]) ** (1 - gamma) * psi[t]


       # a = x_grid[np.argmax(psi_g, axis = 0)].T
    return opt_res(x, 'Discrete grid - no income, risk aversion: ' + str(gamma))


# #smearing_est(y_H_fit, eps_h, 5)
# def root(a, b, c):
#     """
#         Returns the root of a + bx + cx^2 using the abc formula
#     """
#     d = np.sqrt(b ** 2 - 4 * a * c)
#     r = (-b - d) / (2 * a)
#     return

# idx =np.where(np.isnan(d))
# a = x_opt[:, idx]

# idx[8]
     # if polynomial == 1:
     #     #Just select the highest grid point
     #     x_opt = x_grid[np.argmax(psi_g, axis = 0)].T

     # elif polynomial == 2:
     #     z = polyfit(x_grid.squeeze(), psi_g, 2)

     #     x_opt= - z[1, :] / (2 * z[2, :]) #x opt: -b/2c
     #     x_opt = np.maximum(np.minimum(x_opt, 1), 0)
     # elif polynomial == 3:
     #     z = polyfit(x_grid.squeeze(), psi_g, 3)
     #     a = z[1, :]
     #     b = 2 * z[2, :]
     #     c = 3 * z[3, :]
     #     x_opt = root(a, b, c)


     #     x_opt= - z[1, :] / (2 * z[0, :]) #x opt: -b/2c
     #     x_opt = np.maximum(np.minimum(x_opt, 1), 0)
