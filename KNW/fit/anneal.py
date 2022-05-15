# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:37:28 2022

@author: Swen
"""
from scipy.optimize import dual_annealing
import warnings
import numpy as np

def anneal(obj, x0 = None, maxit = 1000, lb = None, ub = None, print_fun = None):
    if lb != None or ub != None:
        bnds = list(zip(lb, ub))
    else:
        bnds = None

    with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          fit = dual_annealing(obj, bounds=bnds, maxiter = maxit, callback = print_fun, x0 = x0, no_local_search = True, initial_temp = 10000)
                  #initial_temp = 1000, no_local_search = False)
    return fit


def anneal2(obj, x0, T0 = 10000, RT = 0.85, Ns = 30, Nt = 30, C = 1.1, maxit = 100, eps = 0.00001, alpha = 0.6, omega = 0.23):
    """
        T0: Initial temperature
        RT: Temperature reduction factor
        Ns: Nr of steps in which only stepsize changes
        Nt: Nr of iterations before temperature is reduced
        maxit: maximum number of iterations
        C: stepsize adjustment
        eps: tolerance of change in obj fun
    """
    #T0 = 10; RT = 0.85; Ns = 30; Nt = 10; maxit = 200; eps = 0.00001; alpha = 0.3; omega = 0.23
    n = len(x0)          #Number of parameters
    D = np.diag(np.ones(n)) /10 #Vector of maximum allowed stepsizes

    x = x0.copy()       #We need a copy, not a reference -> o.w x0 changes as x changes
    f = obj(x)

    T = T0
    x_opt = x.copy()
    f_opt = f.copy()
    f_old = f_opt.copy()

    it = 0
    while it < maxit:
        print('it: ', it, '\tT: ', T, '\tf_opt: ', f_opt)
        print(KNW_pars(x_opt))
        for a in range(Nt):
            u =  np.random.uniform(-1, 1, n)
            np.abs(u)
            R = D @ np.diag(np.abs(u))
            x_can = x + D @u    #Propose candidate
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_can = obj(x_can)
            print('f: ', f)
            print('f_can: ', f_can)
            if f_can < f:       #Accept the solution
                f = f_can
                x = x_can

                D = (1 - alpha) * D + alpha * omega * R
                print('R step: ', np.diagonal(R))
                print('Updated D:', np.diagonal(D))
            else:
                d_avg = np.mean(np.diagonal(D))
                delta_obj = f_can - f
                p_accept = np.exp(- delta_obj / (d_avg * T))

                print('Delta: ', f_can - f)
                print('p_accept: ', p_accept)
                if p_accept > np.random.uniform(0, 1):
                    f = f_can
                    x = x_can
                    D = (1 - alpha) * D + alpha * omega * R
                    print('R step: ', np.diagonal(R))
                    print('Updated D:', np.diagonal(D))
                else:
                    print('Rejected')

            if f_can < f_opt:
                f_opt = f_can
                x_opt = x_can

        #Check convergence:
        if np.linalg.norm(f_old - f_opt) < eps:
            break     #Stop the algorithm

        f_old = f_opt #Keep previous in memory!
        x = x_opt
        T = RT * T
        it = it + 1

    return x_opt, f_opt, V








#### OLD CODE:




#2 --> move in 1 direction

    # while it < maxit:
    #     print('it: ', it, '\tT: ', T, '\tf_opt: ', f_opt)
    #     print(KNW_pars(x_opt))
    #     for a in range(Nt):
    #         n_accept = np.zeros(n)  	#to keep track of accepted
    #         n_try = np.zeros(n)         #to keep track of tried
    #         for b in range(Ns):

    #             for i in range(n):
    #                 x_can = x.copy()                #Set up candidate
    #                 r = np.random.uniform(-1, 1)    #Draw random number
    #                 x_can[i]  = x_can[i] + r * V[i]
    #                 with warnings.catch_warnings():
    #                     warnings.simplefilter("ignore")
    #                     f_can = obj(x_can)
    #                 if f_can <= f:      #new value is better so accept
    #                     x = x_can
    #                     f = f_can
    #                 else:
    #                     p_accept = np.exp(-(f_can - f) / T)
    #                     if p_accept < np.random.uniform(0, 1):
    #                         x = x_can
    #                         f = f_can
    #                         n_accept[i] += 1
    #                     n_try[i] += 1
    #                 if f_can <= f_opt: #Check if we have a new point
    #                     x_opt = x_can
    #                     f_opt = f_can
    #         #Adjust V such that 60% of trials accepted: HOW?
    #         accept_prob = n_accept /n_try
    #         for i in range(n):
    #             print('Elem ', i, '\t V: ', round(V[i], 3), '\t accept_prob: ', round(accept_prob[i], 3))
    #             if accept_prob[i] < 0.6:
    #                 V[i] =  max(V[i] - C, 0.001)   #Increase the stepsize for this par!
    #             elif accept_prob[i] > 0.6: #elif instead of else to handle nan values!
    #                 V[i] = V[i] + C  #Decrease the stepsize for this par!
    #                 #KNW_pars(x_opt)
    #     #Check convergence:
    #     if np.linalg.norm(f_old - f_opt) < eps:
    #         break     #Stop the algorithm

    #     f_old = f_opt #Keep previous in memory!
    #     x = x_opt
    #     T = RT * T
    #     it = it + 1
