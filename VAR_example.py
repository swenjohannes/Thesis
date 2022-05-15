# -*- coding: utf-8 -*-
"""
Created on Sat May  7 18:47:12 2022

@author: Swen

    SINGLE RISKY ASSET PROBLEM

    - model obtained from Brandt et all (2005)
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

############ 5 year with gamma = 5
T = 40# * 4
M = 5000
gamma = 5

[Re, Z] = var.simulate(T,  M, seed = 1710)  #Simulate values

plt.plot(Re)
#percentile_plot(Re)

np.mean(Re) + 1 - rf

#Set correct dimensions!
Re = Re.reshape((1, T +1, M))
Z = Z.reshape(1, T + 1, M)
Rf = np.full((T + 1, M), rf)  #constant, so everywhere the same

#Transform returns:
Re = rf * (np.exp(Re) - 1)



#Discrete grid
discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False]).plot()

#Taylor approx
taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False]).plot()


"""
    Reproducing the results of Kong and Oosterlee 2016:
        - DSG looks quite similar
        - gamma = 15 is a bit too low -> starting @ T = 40
        - 2nd order taylor approx is too low in general, but more stable
        - Stock allocation decreases as risk aversion increases
        - Stock allocation decreases as horizon becomes smaller
"""


T_val = [10, 20, 30, 40]
gamma_val = [5, 15]

m = len(T_val)
n = len(gamma_val)
p = 10
d_grid = np.zeros((m, n, p))
taylor = np.zeros((m, n, p))
for t in range(m):
    for g in range(n):
        for i in range(p):
            #T = 80; gamma = 5
            T = T_val[t]
            gamma = gamma_val[g]

            [Re, Z] = var.simulate(T,  M)  #Simulate values
            #Set correct dimensions!
            Re = Re.reshape((1, T +1, M))
            Z = Z.reshape(1, T + 1, M)
            Rf = np.full((T + 1, M), rf)  #constant, so everywhere the same
            Re = rf * (np.exp(Re) - 1)
            res = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50)
            d_grid[t, g, i] = res.x_mean[0]

            res = taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False])
            taylor[t, g, i] = res.x_mean[0]

np.mean(d_grid, axis = 2)
np.mean(taylor, axis = 2)

#Plots for different gamma's
T = 40
for gamma in [2, 5, 7, 10]:
    discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()
    taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False]).plot()



gamma = 2
res = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False])
#res.x_mean

"""
    Compare certainty equivalents
    - There is added value by using the dynamic strategy.
    - CEV of optimal is 1.38 all others are lower< 1.34
"""

"""
    Now with income
"""
#Simulate wages
mu_l = 0.02
eps = np.random.normal(0, 0.00, size = (T, M))  #Generate random normal
kappa = 0.24

L = np.ones((T + 1, M))
for t in range(T):
    L[t + 1] = L[t] * np.exp(mu_l + eps[t])

taylor_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 25).plot()
discrete_grid_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()



### Varying gamma
for gamma in [2, 5, 7, 10]:
    discrete_grid_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()

### Varying sigma_eps
gamma = 5
for sigma_eps in [0.02, 0.04, 0.06]:
    eps = np.random.normal(0, sigma_eps, size = (T, M))  #Generate random normal
    discrete_grid_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()

### Varying mu_L
eps = np.random.normal(0, 0.02, size = (T, M))  #Generate random normal
for mu_l in [0.01, 0.02, 0.03, 0.04]:
    discrete_grid_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()

### Varying kappa
for kappa in [0.06, 0.12, 0.18]:
    discrete_grid_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()
    discrete_grid_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 20, W_initial = 2).plot()

kappa = 0.05
discrete_grid_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()


### Varying gamma
for gamma in [2, 5, 7, 10]:
    taylor_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()


### Varying mu_L
eps = np.random.normal(0, 0.02, size = (T, M))  #Generate random normal
for mu_l in [0.01, 0.02, 0.03, 0.04]:
    taylor_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()


""" Investigate T and gamma high """

############ 5 year with gamma = 5
T = 20# * 4
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

#Discrete grid
res1 = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], transform = True)
res2 = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False])

res1.plot()
res2.plot()
cev1, WT1 = cev(res1.x_opt, Re, Rf, gamma)
U1 = np.mean(crra(WT1, gamma))
print(U1)


plt.figure()
plt.plot(res1.x_mean)
plt.plot(res2.x_mean)
plt.title("Transformed vs non transformed")
plt.xlabel('Time to maturity')
plt.ylabel('Stock weight')
plt.legend(['Transformed', 'No transform'])

#Taylor approx
res1 = taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], transform = True)
res2 = taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False])
res1.plot()
res2.plot()

res1.x_opt



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
res1 = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 100, transform = True)
res2 = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 100, use_test_port = True, test_ratio = 0.1, transform = True)
res1.plot()
res2.plot()

res1.x_mean
res2.x_mean

plt.plot(res1.x_mean)
plt.plot(res2.x_mean)
