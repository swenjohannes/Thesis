# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:29:19 2022

@author: Swen
"""

def annuity(r, N, A, I = 0):
    """
        r = riskfree rate
        N = number of periods (vector allowed)
        A = Periodic deposit
        I = Initial deposit
    """
    #r = 0.03; N = np.linspace(0, 3, 3); A = 1; I = 1;
    c = (1 + r) ** N - 1
    return I * (1 + r) ** N + A * ( c/r + c)
