# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:35:31 2022

@author: Swen
"""

from numba import njit
import numpy as np

@njit
def L1(X, A, B, yields, sigma_v, tau):
    """
        A:      6x1 params for different maturities
        B:      6x2 params for different maturities
        yields: 6xT vector of yields
    """
    T = len(yields)
    #import matplotlib.pyplot as plt
    #plt.plot(X)

    #Other yields are observed with a meassurement error:
    idx = np.array([0, 2, 4])
    y = yields[:, idx]
    a = A[idx]
    b = B[:, idx]
    v =   y + ( a + X @ b) / tau[idx]

    loglik = -T / 2 * np.log(np.linalg.det(sigma_v))  \
       - 1 / 2 * np.sum(np.array([w @ np.linalg.inv(sigma_v) @ w.T for w in v]))
    return loglik


#Compile once!

B = np.array([[0, 3, 5], [1, 4, 6]])
A = np.array([1, 2, 3])
X = np.array([[0, 3], [1, 4], [5, 6]])
sigma_v = np.diag([1, 2, 3, 4])
tau = np.array([1, 2, 3, 4])
yields = np.array([[0, 3, 3], [1, 4, 6], [5, 6, 5]])
L1(X, A, B, yields, sigma_v, tau)

if __name__ == '__main__':
    @timer
    def func():
        for _ in range(10000):
            L1(X, A, B, yields, sigma_v, tau)

    func()
