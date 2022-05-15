# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:41:59 2022

@author: Swen
"""
from numba import njit
import numpy as np

@njit
def alpha(X):
    return [(np.exp(x) - 1) / x if x != 0 else 1 for x in X]

#Run once to compile!
d = np.array([1, 2, 3, 4])
h = 0.1
alpha(d * h)

if __name__ == '__main__':
    @timer
    def func():
        for _ in range(10000):
            alpha( * h)

    func()
