# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:37:18 2022

@author: Swen
"""
from scipy.optimize import minimize, Bounds #, LinearConstraint, NonlinearConstraint
import warnings
#import numpy as np

# def constraint_matrix(con_idx, n):
#     m = len(con_idx)
#     C = np.zeros((m, n))
#     for j in range(m):
#         C[j] = [1 if i in con_idx[j] else 0 for i in range(26)]
#     return C

def local_search(obj, x0 = None, maxiter = 1000, lb = None, ub = None, EM = None):
    """
        obj:        objective function
        x0:         start x
        maxiter:    maximum number of iterations
        lb:         lower bounds on parameters
        ub:         upper bounds on parameters
        EM:         Expected longrun means restriction!
    """
    #EM = EM[0:3]

    bnds = Bounds(lb, ub, keep_feasible = True)


    #c_idx = [[0], [3, 12], [3]]
    #C = constraint_matrix(c_idx, 26)
    #cons = LinearConstraint(C, EM, EM)


    #Add longrun restrictions:
    #How do we put this as a constraint?
    #Does it work at all?
    #M = K.T + Lambda1.T @ Sigma_x
    #BN_5 = np.linalg.inv(M) @ (expm(- M * 5) - np.identity(2)) @ R1


    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = minimize(obj, x0,
                             options = {'disp': True,
                                        'maxiter': maxiter},
                             method = 'trust-constr',
                             #callback = callbackF,
                             #constraints = cons,
                             bounds = bnds)
    return fit
