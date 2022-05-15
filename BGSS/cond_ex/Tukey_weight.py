# -*- coding: utf-8 -*-
"""
Created on Sun May  1 22:03:17 2022

@author: Swen
"""

import numpy as np
from numba import njit

@njit
def Tukey_weight(u, k = 1.547):
    """
        Tukey bisquare loss function
    """
    w = [(1 - (v / k) **2) ** 2 if np.abs(v) < k else 0 for v in u]
    return np.array(w)

#Run once to compile
a = np.random.normal(0, 3, 10)
b = Tukey_weight(a)

# #@timer
# def func():
#     for _ in range(10000):
#         Tukey_weight(a)
# func()
