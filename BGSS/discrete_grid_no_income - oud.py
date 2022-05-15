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


def discrete_grid_no_income(Re, Rf, Z, gamma = 5, G = 50, basis_order = [2, 0, True, False, False], method = 'ols', transform = False, polynomial = 1):

    if np.ndim(Re) != 3:
        raise ValueError('The dimensions of Re are incorrect. Should be 3 dimensional: K assets x T periods x M observations')
    if np.ndim(Z) != 3:
        raise ValueError('The dimensions of Z are incorrect. Should be 3 dimensional: K assets x T periods x M observations')
    if np.ndim(Rf) != 2:
        raise ValueError('The dimensions of Rf are incorrect. Should be 2 dimensional: T periods x M observations')
    #gamma = 15; G = 100; basis_order = None; method = 'ols'; transform = True; polynomial = 1
    K, T, M = Re.shape


    #G = 100 #G = 20
    x_grid, G = create_x_grid(G, K) #Construct the discrete grid

    psi = np.ones((T, M)) #/ (1 - gamma) #First psi = 1
    x = np.zeros((K, T - 1, M))     #Store opt values

    #Solve T -> 2
    for t in reversed(range(1, T)):
        #t = T - 1#t =10 #t = t -1
        print(t)
        basis = polynomial_basis(Z[:, t - 1], Re[:, t - 1], basis_order)
        psi_g = np.zeros((G, M))

        for g in range(G):
            #g = 0
            y =(np.dot(x_grid[g], Re[:, t]) + Rf[t - 1]) **  (1 - gamma) / (1 - gamma) * psi[t]
            y_fit = cond_ex(y, basis, method, transform, gamma = gamma)
            psi_g[g] =  y_fit
            x_opt = x_grid[np.argmax(psi_g, axis = 0)].T


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
