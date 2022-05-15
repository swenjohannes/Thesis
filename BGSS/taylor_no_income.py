# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:11:46 2022

@author: Swen
"""
import numpy as np
from numba import njit, prange

from .utils import create_x_grid, polynomial_basis
from .cond_ex import cond_ex
from .opt_res import opt_res

@njit
def H_b_inv(u, gamma):
    return (1 / (1 - gamma)) * np.log(u)

@njit
def H_b(u, gamma):
    return np.exp(u * (1 - gamma))

@njit(parallel=True)
def smearing_est2(Y, eps, gamma):
    #Y = y_H_fit; eps = eps_h;
    M = len(Y)
    est = np.zeros(M)
    for m in prange(M):
        #m = 1
        est[m] = np.mean(H_b(Y[m] + eps, gamma))
    return est

#Run once to compile
smearing_est2(np.ones((2, 3)), np.ones((2, 3)), 5)



def taylor_no_income(Re, Rf, Z, gamma = 5, G  = 10, basis_order = None, method = 'ols', transform = True):
    if np.ndim(Re) != 3:
        raise ValueError('The dimensions of Re are incorrect. Should be 3 dimensional: K assets x T periods x M observations')
    if np.ndim(Z) != 3:
        raise ValueError('The dimensions of Z are incorrect. Should be 3 dimensional: K assets x T periods x M observations')
    if np.ndim(Rf) != 2:
        raise ValueError('The dimensions of Rf are incorrect. Should be 2 dimensional: T periods x M observations')

    #gamma = 15; G = 10; basis_order = None; method = 'ols'; transform = False
    #Construct the discrete grid
    K, T, M = Re.shape
    x_grid, G = create_x_grid(G, K)

    psi = np.ones((T, M))  #First psi = 1
    x = np.zeros((K, T - 1, M))     #Store opt values
    for t in reversed(range(1, T)):
        #t = T - 1# t = t - 1
        print(t)
        basis = polynomial_basis(Z[:, t - 1, :], Re[:, t - 1, :], basis_order)

       #SOLVE SECOND ORDER APPROXIMATION
        a_hat = np.zeros((M, K)) #Asset wise conditinal expectation
        for i in range(K):
            #i = 0
            y = Rf[t - 1] ** -gamma * Re[i, t] * psi[t]
            a_hat[:, i] = cond_ex(y, basis, method)

            #ols(y, basis).plot()
            #ols(np.log(y **  (1 / (1 - gamma))), basis).plot()
            #ols(y**  (1 / (1 - gamma)), basis).plot()

        b_hat = np.zeros((M, K, K))
        for i in range(K):
            for j in range(K):
                y = Rf[t - 1] ** (-1 -gamma)  * Re[i, t, :] * Re[j, t, :] * psi[t]
                # if transform:
                #      Y_inv = H_b_inv(y, gamma)
                #      Y_inv_hat = cond_ex(Y_inv, basis)
                #      eps_h = Y_inv - Y_inv_hat
                #      y_hat = smearing_est2(Y_inv_hat, eps_h, gamma) #Smearing estimate
                # else:
                y_fit = cond_ex(y, basis, method)
                b_hat[:, i, j] = y_fit

                # ols(Y_inv, basis).plot()

                # plt.figure()
                # plt.scatter(basis[:, 1], y)
                # plt.scatter(basis[:, 1], y_hat)
                # plt.scatter(basis[:, 1], y_fit)
                # plt.legend(['actual', 'transform', 'normal'])

                # plt.figure()
                # plt.scatter(basis[:, 1], y_hat)
                # plt.scatter(basis[:, 1], y_fit)
                # plt.legend(['transform', 'normal'])
                # ols(y, basis).plot()
                # ols(np.log(y ** (1 / (1 - gamma))), basis).plot()
                # ols((1 / (1 - gamma)) * np.log(y), basis).plot()


             #plt.scatter(H_inv(y, gamma), basis[:, 1])

             #plt.scatter(basis[:, 1], np.log(y))

        if K == 1:
            x_opt = a_hat.squeeze() / (gamma * b_hat.squeeze()) #unconstrained minimum
            x_opt = np.maximum(np.minimum(x_opt, 1), 0) #Add constraints
        else:
            y_g = np.zeros((G, M))
            for g in range(G):
                y_g[g] = x_grid[g] @ a_hat.T - 0.5 * gamma * x_grid[g] @ b_hat @ x_grid[g].T

            g_max = np.argmax(y_g, axis = 0)
            x_opt = x_grid[g_max].T

        x[:, t - 1] = x_opt
        psi[t - 1] = psi[t] * (np.sum(x_opt *  Re[:, t], axis = 0) + Rf[t -1]) ** (1 - gamma)
        # a = x_grid[g_max]
        #np.mean(x[:, t -1], axis = 1)
    return opt_res(x, 'Taylor - no income, risk aversion: ' + str(gamma))

#stack_plot(x)

#opt_res(x, 'Taylor - no income, risk aversion: ' + str(gamma)).x_mean
