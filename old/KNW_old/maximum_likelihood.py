# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:34:48 2022

@author: Swen
"""
import numpy as np
from BN import BN
from AN import AN
from alpha import alpha

def maximum_likelihood(x, data):
    """
        Maximum likelihood function

        Input:
            x   Vector of parameters 23 in total
    """
    #x = np.array([i for i in range(28)])
    #x = x0  #x = a # x= x_opt.

    #Extract parameters from x vector!
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

    sigma_v = np.diag(x[23:26])
    sigma_v = sigma_v.astype(float)

    yields = np.array(data)[:, 0:5]

    #Compute A and B

    tau = np.array([1, 2, 3, 5, 10])
    M = K.T + Lambda1.T

    try:
        B = BN(M, tau, R1)
        A = AN(M, tau, 0.1, Lambda0, R0, R1)

        T = len(yields)
        idx = np.array([1, 3]) #2Y and 5Y are observed without error

        #Compute X's
        y = yields[:, idx]
        a = A[idx]
        b = B[:, idx].T  #transpose

        l3 = -0.5 * T * np.log(np.linalg.det(b))
        X = -( y * tau[idx] + a) @ np.linalg.inv(b)

        #plt.plot(yields)
        #plt.plot(X) #NOTE: x's are completely off...l B calculation is wrong!

        idx = np.array([0, 2, 4])
        y = yields[:, idx]
        a = A[idx]
        b = B[:, idx]
        v =   y + ( a + X @ b) / tau[idx]

        l1 = -T / 2 * np.log(np.linalg.det(sigma_v))  \
             - 1 / 2 * np.sum(np.array([w @ np.linalg.inv(sigma_v) @ w.T for w in v]))

        """ Determine L2"""

        Y = np.concatenate([X, data[[ 'Inflation','^RUI']]], axis = 1)
        theta0 = np.array([0,
                           0,
                           delta_0pi - 0.5 * sigma_pi @ sigma_pi.T,
                           R0 + eta_s - 0.5 * sigma_s @ sigma_s.T])
        theta1 = np.array([-K[0],
                           -K[1],
                           delta_1pi.T,
                           R1.T])

        theta1 = np.concatenate((theta1, np.zeros((4,2))), axis =1)

        Sigma_y= np.concatenate((np.identity(2), np.zeros((2,2))), axis = 1)
        Sigma_y = np.concatenate([Sigma_y, sigma_pi.reshape(1, 4), sigma_s.reshape(1, 4)])

        d, U = np.linalg.eig(theta1)

        U_inv = np.linalg.inv(U)

        h = 1/12
        gamma_h = U @ np.diag(np.exp(d * h )) @ U_inv
        F = h * np.diag( alpha(d * h))
        mu_h = U @ F @ U_inv @ theta0


        inner = np.add.outer(d, d) * h
        #b = np.array([alpha(v) for v in inner])
        V = U_inv @ Sigma_y @ Sigma_y.T @ U_inv.T * h * np.array([alpha(v) for v in inner])
        sigma_h = U @ V @ U.T


        """  IMPORTANT: Y is length 220. This one misses 1!"""
        #b = Y[:-1] @ gamma_h
        #c = np.linalg.inv(sigma_h)
        epsilon = Y[1:] - mu_h  -  Y[:-1] @ gamma_h

        l2 = -0.5 * T * np.linalg.det(sigma_h) \
            - 0.5 * np.sum([eps @ np.linalg.inv(sigma_h) @ eps.T for eps in epsilon])

        loglik = l1 + l2 + l3
    except:
        loglik = -np.nan #A high penalty in case something fails!
    if np.isnan(loglik):
        loglik = -np.nan #L3 sometimes causes this error.
    return loglik



# x0 = np.array([0.0181, -0.0063, 0.0014, 0.0240, -0.0148, 0.0053, 0.08, -0.19, 0.35,
#       0.0002, -0.0001, 0.0061,  0.0452, -0.0053, -0.0076, -0.0211, 0.1659,
#       0.403, 0.039, 0.149, -0.381, 0.089, -0.083, 1, 1, 1])

# maximum_likelihood(x0, data)

# @timer
# def func():
#     for _ in range(1000):
#         maximum_likelihood(x0, data)
# func()
