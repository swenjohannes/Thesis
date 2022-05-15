# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:44:21 2022

@author: Swen
"""

import numpy as np

from .utils import create_x_grid, create_W_grid, polynomial_basis, Chi, Psi, interpolate_x
from .cond_ex import cond_ex
from .opt_res import opt_res

def taylor_income(Re, Rf, Z, mu_l , eps, kappa, gamma = 5, G  = 10, J = 11, d = 0.15, basis_order = None, method = 'ols'):
    #gamma =7; G = 5; basis_order = None; method = 'ols'; kappa = 0.12; J = 10; d = 0.1;
    if np.ndim(Re) != 3:
        raise ValueError('The dimensions of Re are incorrect. Should be 3 dimensional: K assets x T periods x M observations')
    if np.ndim(Z) != 3:
        raise ValueError('The dimensions of Z are incorrect. Should be 3 dimensional: K assets x T periods x M observations')
    if np.ndim(Rf) != 2:
        raise ValueError('The dimensions of Rf are incorrect. Should be 2 dimensional: T periods x M observations')

    K, T, M = Re.shape
    rf = np.mean(Rf[0])              #Used to calculate annuity!
    T = T - 1

    W_grid = create_W_grid(kappa, rf, T, J, d) #Construct W grid
    x_grid, G = create_x_grid(G, K) #Contruct X grid

    x = np.zeros((J, K, T, M))          #Store opt values

    ###### SOLVE FIRST PERIOD #################
    t = T
    print(t)
    basis = polynomial_basis(Z[:, t - 1, :], Re[:, t - 1, :], basis_order)

    for j in range(J):
        #j = 1
        W_hat = (W_grid[t - 1, j] + kappa) * Rf[t - 1]

        a_hat = np.zeros((M, K)) #Asset wise conditinal expectation
        for k in range(K):
            a_hat[:, k] = cond_ex(W_hat ** -gamma * Re[k, t], basis)

        b_hat = np.zeros((M, K, K))
        for p in range(K):
            for q in range(K):
                 b_hat[:, p, q] = cond_ex(W_hat ** (-gamma - 1) * Re[p, t, :] * Re[q, t, :], basis)

        y_g = np.zeros((G, M))
        for g in range(G):
             #g = 8
             y_g[g] = x_grid[g] @ a_hat.T - \
                 0.5 * gamma * x_grid[g] @ b_hat @ x_grid[g].T * (W_grid[t - 1, j] + kappa)

        x[j, :, t - 1] = x_grid[np.argmax(y_g, axis = 0)].T

    #np.mean(x[:, :, t - 1], axis = 2)

    ###### SOLVE T -1 -> t  #################
    for t in reversed(range(1, T)):
    #for t in reversed(range(2, T)):
        #t = 1 #t = T -1 #t = t -1
        print(t)
        basis = polynomial_basis(Z[:, t - 1, :], Re[:, t - 1, :], basis_order)

        #gridpoints = J if t > 1 else 1 #Only check 1 grid point in the last period!
        used_x = np.zeros((J, K, T - t, M)) #Matrix to store the used optimal x_hat.

        for j in range(J):
            #j = 20
            Wtp1 = (W_grid[t - 1, j] + kappa) * Rf[t - 1]

            #a = Re[:, t:T]
            #Interpolate Wtp1 so we obtain the optimal sequence x_star
            x_hat  = interpolate_x(Wtp1, W_grid[t], x[:, :, t:T, :])
            psi_hat = Psi(x_hat, Re[:, (t + 1):(T + 1)], Rf[t:T]) #Compute phi_hat from x_hat
            chi_hat = Chi(psi_hat, eps[t:T], mu_l) #Compute chi_hat from x_hat

            W_hat = Wtp1 * psi_hat[0] + kappa * chi_hat #Compute terminal wealth

            a_hat = np.zeros((M, K)) #Asset wise conditinal expectation
            for i in range(K):
                #i = 0
                a_hat[:, i] = cond_ex(W_hat.squeeze() ** -gamma * Re[i, t] * psi_hat[0], basis)

            #res = ols(W_hat.squeeze() ** -gamma * Re[0, t], basis)
            #res.plot()
            #plt.plot(a_hat)
            #plt.legend(['stock', 'bond'])


            b_hat = np.zeros((M, K, K))
            for p in range(K):
                for q in range(K):
                    b_hat[:, p, q] = cond_ex(W_hat.squeeze()** (-1 -gamma)  * Re[p, t, :] * Re[q, t, :] * psi_hat[0] **2,  basis)

            #Unconstrained optimum
            # x_opt = np.zeros((M, 2))
            # for m in range(M):
            #     x_opt[m] = 1/ (gamma * (W_grid[t - 1, j] + kappa))  * np.linalg.inv(b_hat[m]) @ a_hat[m]


            y_g = np.zeros((G, M))
            for g in range(G):
                y_g[g] = x_grid[g] @ a_hat.T - \
                    0.5 * gamma * x_grid[g] @ b_hat @ x_grid[g].T * (W_grid[t - 1, j] + kappa)

            #x_opt =x_grid[np.argmax(y_g, axis = 0)].T
            g_idx = np.argmax(y_g, axis = 0)
            x_opt = x_grid[g_idx].T

            x[j, :, t - 1] = x_opt

            #a = x_grid[g_idx].T
            #print(np.mean(a, axis = 1))

            for m in range(M):
                used_x[j, :, : , :] = x_hat #Store X_hat that was used!

        x[:, :, t:T, :] = used_x
        #a = np.mean(x[0, :, t:T])
        #a = np.mean(x[10, :, t:T])
        #stack_plot(x[0, :, t:T])
        #stack_plot(x[10, :, t:T])
        #a = used_x[4]
    # wealth grid has zero influence
    #for j in range(J):
    #   stack_plot(x[j])

    return opt_res(x[0], 'Taylor - with income, risk aversion: ' + str(gamma))  #Only return the value of the first grid point = 0

#stack_plot(x[0])
