# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:37:11 2022

@author: Swen
"""

from numba import njit
import numpy as np

@njit(cache = True)
def expm(A):
    d, U = np.linalg.eig(A)
    return U @ np.diag(np.exp(d)) @ np.linalg.inv(U)

#Run once to compile!
expm(-np.ones((2,2)))

if __name__ == '__main__':
    #Check wherther answer is correct:
    M = np.ones((2,2))
    import scipy
    from Timer.Timer import timer
    scipy.linalg.expm(-M)

    @timer
    def func():
        for _ in range(10000):
            expm(-M)
    func()
