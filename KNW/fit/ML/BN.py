# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:01:22 2022

@author: Swen
"""

from numba import njit
import numpy as np
#import expm
from .expm import expm

#USE THESE IF RUNNING FROM HERE:
# from expm import expm

@njit
def BN(M, tau, R1):
    #IMPORTANT: CHECK THIS FORMULA, MIGHT BE WRONG
    n = len(tau)
    B = np.zeros((2, n))
    Minv = np.linalg.inv(M)
    for t in range(n):
        B[:, t] = Minv @ (expm(-M  * tau[t]) - np.identity(2)) @ R1
    return(B)

#Compile once:
M =  np.array([[0.4, 0.6], [0.1, 0.2]])
tau =  np.array(np.arange(0, 5, 1))
R1 =  np.array([0.2, 0.3]).T

BN(M, tau, R1)

""" TEST SECTION """
if __name__ == '__main__':
    @timer
    def func():
        for _ in range(10000):
            BN(M, tau, R1)
    func()
