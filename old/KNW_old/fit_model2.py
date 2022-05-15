# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:42:43 2022

@author: Swen
"""
from scipy.optimize import minimize, dual_annealing, Bounds
from maximum_likelihood import maximum_likelihood
import numpy as np
import warnings

def extract_x(x):
    delta_0pi = x[0]
    delta_1pi = x[[1, 2]]

    R0 = x[3]
    R1 = x[[4, 5]]

    K = np.zeros([2, 2])
    K[0, 0] = x[6]
    K[1, 0] = x[7]
    K[1, 1] = x[8]

    sigma_pi = np.concatenate([x[9:12], np.zeros(1)])
    eta_s = x[12]
    sigma_s = x[13:17]

    Lambda0 = x[17:19]
    Lambda1 = x[19:23].reshape(2,2)

    sigma_v = x[23:26]
    return (delta_0pi, delta_1pi, R0, R1, K, sigma_pi, eta_s,
            sigma_s, Lambda0, Lambda1, sigma_v)

def print_x(x):
    (delta_0pi, delta_1pi, R0, R1, K,
     sigma_pi, eta_s, sigma_s, Lambda0, Lambda1, sigma_v) = extract_x(x)

    print("\t pi0: \t", delta_0pi)
    print("\t pi1: \t", delta_1pi)
    print("\t R0: \t", R0)
    print("\t R1: \t", R1)
    print("\t K: \t", K[0])
    print("\t \t \t", K[1])
    print("\t s_pi: \t", sigma_pi)
    print("\t s_s: \t", sigma_s)
    print("\t eta: \t", eta_s)
    print("\t L0: \t", Lambda0)
    print("\t L1: \t", Lambda1[0])
    print("\t \t \t", Lambda1[1])
    print("\t s_v: \t", sigma_v)

def callbackF(x, state):
    print_x(x)

def print_fun(x, f, accepted):
         print("\r\r\r\r\r\r\r\r\r\r\r\r\r\r at minima %.4f accepted %d" % (f, int(accepted)))
         #x = lb
         #Extract parameters from x vector!
         print_x(x)

lb =[-0.01,#delta_pi0
         -0.05, -0.05, #delta_pi1
         -0.01, #R0
         -0.05, -0.05, #R1
         -1, -1, -1, #K
         -0.1, -0.1, -0.1, #sigma_pi
         0.02, #eta_s
         -0.1, -0.1, -0.2, 0.1, #sigma_s
         -1, -1, #Lambda0
         -1, -1, -1, -1,#Lambda1
         1e-6, 1e-6, 1e-6]

ub = [0.03,#delta_pi0
      0.05, 0.05, #delta_pi1
      0.05, #R0
      0.05, 0.05, #R1
      3, 3, 3, #K
      0.1, 0.1, 0.2, #sigma_pi
      0.08, #eta_s
      0.1, 0.1, 0.1, 0.3, #sigma_s
      2, 2, #Lambda0
      2, 2, 2, 2,#Lambda1
      0.1, 0.1, 0.1] #sigma_v

#PRINT IF REQUESTED:
# print("Lower bounds:")
# print_x(np.array(lb))
# print("Upper bounds:")
# print_x(np.array(ub))

x0 = np.array([0.0181, -0.0063, 0.0014, 0.0240, -0.0148, 0.0053, 0.08, -0.19, 0.35,
      0.0002, -0.0001, 0.0061,  0.0452, -0.0053, -0.0076, -0.0211, 0.1659,
      0.403, 0.039, 0.149, -0.381, 0.089, -0.083, 0.01, 0.01, 0.01])

x0 = np.array([ 0.01809023, -0.00629713,  0.0013914 ,  0.02118738, -0.02037945,
       -0.00043192,  0.06040135, -0.18654586,  0.31844334,  0.00129675,
       -0.00114124,  0.09803317,  0.04519558, -0.00524736, -0.00763624,
       -0.02051171,  0.16622621,  0.39602062,  0.03799107,  0.13299393,
       -0.41369938,  0.09156032, -0.10366804,  0.01108664,  0.01008853,
        0.0091812 ])

print_x(x0)

def fit_model(data, x0 = None, maxiter = 1000, method = 'local'):
    """
        Fits the model to the input data

        Returns:
            fitted model
    """
    #maxiter = 1000
    obj = lambda x: -maximum_likelihood(x, data)

    print("Start value:")
    print_x(x0)
    print(obj(x0))

    x0 = x_opt
    if method == 'local':
        fit = local_search(obj, x0, 1000, lb, ub) #use local search
    elif method == 'anneal':
        fit = anneal(obj, x0, 10000, lb, ub) #use simulated annealation

    x_opt = fit.x
    print("Optimal value")
    print(obj(x_opt))
    print_x(x_opt)

    #print(np.where(ub - x_opt < 0.0001))
    #print(np.where(x_opt - lb < 0.0001))
    return x_opt

def local_search(obj, x0, maxiter, lb, ub):
    bnds = Bounds(lb, ub, keep_feasible = True)
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = minimize(obj, x0,
                             options = {'disp': True,
                                        'maxiter': maxiter},
                             method = 'trust-constr',
                             #callback = callbackF,
                             bounds = bnds)
    return fit

def anneal(obj, x0, maxiter, lb, ub):
    bnds = list(zip(lb, ub))
    with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          fit = dual_annealing(obj, bounds=bnds, maxiter = maxiter, callback = print_fun)
    return fit


#x_local = x_opt
#x_anneal = x_opt
print_x(x_local)

delta_0pi, delta_1pi, R0, R1, K, sigma_pi, eta_s, sigma_s, Lambda0, Lambda1, sigma_v = extract_x(x_local)
