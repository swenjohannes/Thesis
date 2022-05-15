# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:05:06 2022

@author: Swen
"""

#from numba import njit
import numpy as np
from .BN import BN

#USE THESE IF RUNNING FROM HERE:
# from BN import BN

#@njit
def AN(M, tau, Lambda0, R0, R1, delta =0.1):
    """
        M = pricing kernel
        tau = tau's to compute
        delta = integration stepsize

        Note: This can probably be done analytically!!
    """
    #delta = 0.1
    delta = np.array(delta)
    T = tau[-1]
    Tau = np.arange(np.array(0), np.array(T), delta)

    B = BN(M, Tau, R1)
    B * B
    x = B.T @ B
    An_dot = -R0 -Lambda0.T @ B + 0.5 * np.sum(B * B, axis = 0)

    idx = tau // delta
    idx = idx.astype(int)
    n = len(tau)
    A = np.zeros(n)
    for t in range(n):
        A[t] = np.trapz(An_dot[:idx[t]], dx=delta) #Integrate y

    return A

#Compile once:
M =  np.array([[0.4, 0.6], [0.1, 0.2]])
tau =  np.array(np.arange(0, 5, 1))
R0 =  np.array([0.1])
R1 =  np.array([0.2, 0.3]).T
Lambda0 = np.array([0.4, 0.1])
A = AN(M, tau, Lambda0, R0, R1)

""" TEST SECTION """
if __name__ == '__main__':
    @timer
    def func():
        for _ in range(10000):
            AN(M, tau, 0.1, Lambda0, R0, R1)
    func()
