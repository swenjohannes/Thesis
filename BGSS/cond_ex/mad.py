# -*- coding: utf-8 -*-
"""
Created on Sun May  1 22:02:36 2022

@author: Swen
"""
import numpy as np
from numba import njit

@njit
def mad(error):
    """
        Returns the mean absolute deviation: mean(abs(e) - mean(e)))
    """
    return np.mean(np.absolute(error) - np.mean(error)) #mean absolute deviation

#Run once to compile
mad(np.random.normal(5, 2, 10))
