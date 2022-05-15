# -*- coding: utf-8 -*-
"""
Created on Sun May 15 17:26:57 2022

@author: Swen
"""

import numpy as np
import matplotlib.pyplot as plt

from BGSS import *
from VAR import VAR
from KNW import *

mu =     [0.227, -0.155]
ar =    [[0, 0.060],
         [0, 0.958]]
sigma = [[0.006, -0.0051],
         [-0.0051, 0.0049]]

y0 = [0, -0.155 / (1 - 0.958)]  #Start conditions VAR
rf = (1.06) ** 0.25             #Risk free rate
gamma = 5                       #Risk aversion parameter

var = VAR(mu, sigma, ar, y0)    #Create VAR Object

""" Test poly fit """
T = 40# * 4
M = 10000
#gamma = 5
gamma = 15
#gamma = 15

[Re, Z] = var.simulate(T,  M, seed = 1710)  #Simulate values

#plt.plot(Re)
#percentile_plot(Re)

np.mean(Re) + 1 - rf

#Set correct dimensions!
Re = Re.reshape((1, T +1, M))
Z = Z.reshape(1, T + 1, M)
Rf = np.full((T + 1, M), rf)  #constant, so everywhere the same

#Transform returns:
Re = rf * (np.exp(Re) - 1)

""" TEST POLYFIT """
#res1 = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 100, transform = True)
res2 = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 100, use_test_port = True, test_ratio = 0.1, transform = True)
#res1.plot()
res2.plot()

res1.x_mean
res2.x_mean

plt.plot(res1.x_mean)
plt.plot(res2.x_mean)
