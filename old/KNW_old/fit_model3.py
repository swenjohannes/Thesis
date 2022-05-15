# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 00:17:55 2022

@author: Swen
"""
from scipy.optimize import minimize, dual_annealing, Bounds
from maximum_likelihood import maximum_likelihood
import numpy as np
import warnings
from KNW_pars import KNW_pars

def fit_model(data, x0, lb, ub, maxiter = 1000, method = 'local'):
    """
        Fits the model to the input data

        Returns:
            - optimal parameters as KNW_pars object
    """
    #maxiter = 1000
    obj = lambda x: -maximum_likelihood(x, data)

    if method == 'local':
        fit = local_search(obj, x0, maxiter, lb, ub) #use local search
    elif method == 'anneal':
        fit = anneal(obj, x0, maxiter, lb, ub) #use simulated annealation

    pars = KNW_pars(fit.x) #Transfer into param object!
    print("\n Maximumlikelihood: ", maximum_likelihood(pars.x, data))
    print("Obtained solution: \n", pars)
    return pars

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

def print_fun(x, f, accepted):
    print("\n Objective: ", f)
    print(KNW_pars(x))




#### TEST SECTION:
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


x0 = np.array([0.0181, -0.0063, 0.0014, 0.0240, -0.0148, 0.0053, 0.08, -0.19, 0.35,
       0.0002, -0.0001, 0.0061,  0.0452, -0.0053, -0.0076, -0.0211, 0.1659,
       0.403, 0.039, 0.149, -0.381, 0.089, -0.083, 0.01, 0.01, 0.01])

x0 = np.array([ 0.01809023, -0.00629713,  0.0013914 ,  0.02118738, -0.02037945,
       -0.00043192,  0.06040135, -0.18654586,  0.31844334,  0.00129675,
       -0.00114124,  0.09803317,  0.04519558, -0.00524736, -0.00763624,
       -0.02051171,  0.16622621,  0.39602062,  0.03799107,  0.13299393,
       -0.41369938,  0.09156032, -0.10366804,  0.01108664,  0.01008853,
        0.0091812 ])

pars_0 = KNW_pars(x0)
print(pars_0)

#Fit local
fit_local = fit_model(data, pars_0.x, lb, ub)
#Fit dual-anneal
fit_anneal = fit_model(data, pars_0.x, lb, ub,
                       maxiter =10, method = 'anneal')
