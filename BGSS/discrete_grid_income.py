# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:11:46 2022

@author: Swen
"""
import numpy as np

from .utils import create_x_grid, create_W_grid, polynomial_basis, Chi, Psi, interpolate_x
from .cond_ex import cond_ex, fast_ols
from .opt_res import opt_res


def discrete_grid_income(Re, Rf, Z, mu_l , eps, kappa, gamma = 5, G  = 10, J = 11, d = 0.2, W_init = 0, basis_order = None, interpolate = True):
    #kappa = 0.1; gamma =  5; G = 5; J = 21; d = 0.2; basis_order = [2, 0, True, False, False]; mu_l; W_init = 0

    if np.ndim(Re) != 3:
        raise ValueError('The dimensions of Re are incorrect. Should be 3 dimensional: K assets x T periods x M observations')
    if np.ndim(Z) != 3:
        raise ValueError('The dimensions of Z are incorrect. Should be 3 dimensional: K assets x T periods x M observations')
    if np.ndim(Rf) != 2:
        raise ValueError('The dimensions of Rf are incorrect. Should be 2 dimensional: T periods x M observations')

    #kappa = 0.1
    K, T, M = Re.shape
    rf = np.mean(Rf[0])              #Used to calculate annuity!
    T = T - 1

    W_grid = create_W_grid(kappa, rf, T, J, d, W_init) #Construct W grid


    #W_grid = np.ones(T) #When running W === 1, we get similar results as first.
    #minmax = np.linspace(1 - d, 1 + d, J) * rf
    #W_grid = np.outer(W_grid, minmax)
    x_grid, G = create_x_grid(G, K) #Contruct X grid

    x = np.zeros((J, K, T, M))          #Store opt values

    ###### SOLVE FIRST PERIOD #################
    t = T
    print(t)
    basis = polynomial_basis(Z[:, t - 1, :], Re[:, t - 1, :], basis_order)
    for j in range(J):
        W_tp1 = W_grid[t - 1, j] + kappa
        psi_g = np.zeros((G, M))
        for g in range(G):
            psi_g[g] = cond_ex((W_tp1 * (np.dot(x_grid[g], Re[:, t]) + Rf[t -1])) \
                            ** (1-gamma) / (1 - gamma)  , basis)

        x_opt = x_grid[np.argmax(psi_g, axis = 0)].T
        x[j, :, t - 1] = x_opt

    print(np.mean(x[:, :, t - 1], axis = 2)) #all equal, all good
    #np.mean(x_opt, axis = 1)
    #np.mean(x[:, :, t - 1], axis = 2) #all equal, all good
    #np.mean(x_opt / L[t], axis = 1 )

    ###### SOLVE T -1 -> t  #################
    for t in reversed(range(1, T)):
    #for t in reversed(range(2, T)):
        #t = T -1 #t = t -1 #t = 1
        print(t)
        basis = polynomial_basis(Z[:, t - 1, :], Re[:, t - 1, :], basis_order)

        #gridpoints = J if t > 1 else 1 #Only check 1 grid point in the last period!
        used_x = np.zeros((J, K, T - t, M)) #Matrix to store the used optimal x_hat.
        #for j in range(gridpoints):
        for j in range(J):
            #j = 0
            psi_g = np.zeros((G, M))
            x_hat = np.zeros((G, K, T - t, M))
            for g in range(G):
                #g =0
                Wtp1 = (W_grid[t - 1, j] + kappa) * (np.dot(x_grid[g], Re[:, t]) + Rf[t - 1])

                #a = interpolate_x(Wtp1, W_grid[t], x[:, :, t:T])

                x_hat[g] = interpolate_x(Wtp1, W_grid[t], x[:, :, t:T]) #Interpolate x* from Wtp1



                #x_hat[g] = x[j, :, t:(T+1)]
                #TEST chi/psi : OK
                #a = np.cumprod(( x_hat[g] * Re[:, t+1:(T + 1)] + Rf[t:T:T]), axis = 1)
                #np.exp(mu_l + eps[t]) *( x_hat[g] * Re[:, t+1:(T + 1)] + Rf[t:T:T])
                #np.exp(mu_l + eps[t]) * psi_hat[0] + np.exp(2 * mu_l + eps[t] + eps[t + 1]) * psi_hat[1]
                #np.exp(mu_l + eps[t + 1]) * psi_hat[1]
                psi_hat = Psi(x_hat[g], Re[:, (t + 1):(T + 1)], Rf[t:T]) #Compute phi_hat from x_hat
                chi_hat = Chi(psi_hat, eps[t:T], mu_l)               #Compute chi_hat from x_hat

                W_hat = Wtp1 * psi_hat[0] + kappa * chi_hat
                #a = W_hat[basis[:, 1].argsort()]** (1 - gamma) / (1 - gamma)
                #Compute expected value for gridpoint

                psi_g[g] = cond_ex(W_hat.squeeze() ** (1 - gamma) / (1 - gamma), basis) #, True)

                #ols(W_hat.squeeze() ** (1 - gamma) / (1 - gamma), basis).plot()
            #psi_g / (1 - gamma)
            g_idx = np.argmax(psi_g, axis = 0)
            #a = x_grid[g_idx].T
            #print(np.mean(a, axis = 1))
            x_opt = x_grid[g_idx].T  #Scale back to original parameters
            #np.mean(x_opt, axis = 1)
            #np.mean(x_opt / L[t], axis = 1)
            x[j, :, t - 1] = x_opt


            for m in range(M):
                #m = 1
                used_x[j, :, : , m] = x_hat[g_idx[m], :, : , m]  #Store X_hat that was used!

        print(np.mean(x[:, :, t - 1], axis = 2)) #all equal, all good
        #a = x_hat[:, :, 0, :]
        x[:, :, t:T, :] = used_x

        #for j in range(J):
        #    stack_plot(used_x[j], j)
        #for j in range(J):
         #   stack_plot(x[j], j)
    #a = x[:, :, 0, :]
    #stack_plot(x[5, :, (t-1):T, :], T - t, "Income - Discrete grid")   #Display the results

    return opt_res(x[0], 'Discrete grid - with income, risk aversion: ' + str(gamma))  #Only return the value of the first grid point = 0

    """ WEALTH GRID PLOT"""
    # T = 15
    # W_grid = create_W_grid(kappa, rf, T, J, d, W_init) #Construct W grid
    # plt.plot(W_grid)
    # plt.title("Wealth grid points")








    #for i in range(10):
        #opt_res(x[i], 'Discrete grid - no income, risk aversion: ' + str(gamma)).plot()
    # ###### SOLVE LAST PERIOD  #################
    # t = 1
    # print(t)
    # J_g = np.zeros((G, M))
    # x_hat = np.zeros((G, K, T - t, M))
    # for g in range(G):
    #     Wtp1 = kappa * (np.dot(x_grid[g], Re[:, t]) + Rf[t -1])

    #     x_hat[g] = interpolate_x(Wtp1, W_grid[t], x[:, :, t:T]) #Interpolate x* from Wtp1
    #     psi_hat = Psi(x_hat[g], Re[:, t:T], Rf[t:T])           #Compute phi_hat from x_hat
    #     chi_hat = Chi(psi_hat, eps[t:T], mu_l)               #Compute chi_hat from x_hat

    #     W_hat = Wtp1 * psi_hat[0] + kappa * chi_hat
    #     J_g[g] = W_hat ** (1 - gamma) / (1 - gamma)

    # #DETERMINE FINAL SOLUTION:
    # g_idx = np.argmax(J_g, axis = 0)
    # x_opt = np.zeros((K, T, M))
    # x_opt[:, 0, :] = x_grid[g_idx].reshape(K, M)

    # for m in range(M):
    #     x_opt[:, 1:, m] = x_hat[g_idx[m], :, :, m]
