# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:19:28 2022

@author: Swen
"""
import numpy as np
import matplotlib.pyplot as plt

from BGSS import *
from VAR import VAR
from Utils import *

def crra(W, gamma):
    return W ** (1 - gamma) / (1 - gamma)

def cev(x_opt, Re, Rf, gamma):
    #x_opt = res.x_opt
    # = np.sum(x_opt * Re[:, 1:], axis = 0) + Rf[:-1]
    outcomes = np.sum(x_opt * Re[:, 1:], axis = 0) + Rf[:-1]
    outcomes = np.concatenate([np.full((1, M), 1), outcomes], axis = 0)
    W_T = np.prod(outcomes, axis = 0)
    U = crra(W_T, gamma)
    cev = (np.mean(U) * (1 - gamma)) ** (1/ (1 - gamma))
    print(cev)
    return cev, W_T


def cev_income(x_opt, Re, Rf, kappa, L, gamma, W_init = 0):
    #x_opt = res2.x_opt;  W_init = 0
    #outcomes =  x_opt * Re[0, 1:] + Rf[:-1]
    outcomes = np.sum(x_opt * Re[:, 1:], axis = 0) + Rf[:-1]
    premium = L * kappa
    W_T = W_init
    for i in range(len(outcomes)):
        W_T = (W_T + premium[i]) * outcomes[i]

    #np.mean(W_T)
    #plt.hist(W_T, bins = 100)
    U = crra(W_T, gamma)
    #print('Mean utility:', np.mean(U), ' vol: ', vol(U))

    #np.mean(U)
    cev = (np.mean(U) * (1 - gamma)) ** (1/ (1 - gamma))
    print(cev)
    return cev, W_T
"""
    Simulation
"""
T = 80# * 4
M = 10000
gamma = 5



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


[Re, Z] = var.simulate(T,  M, seed = 1710)  #Simulate values

#Set correct dimensions!
Re = Re.reshape((1, T +1, M))
Z = Z.reshape(1, T + 1, M)
Rf = np.full((T + 1, M), rf)  #constant, so everywhere the same

#Transform returns:
Re = rf * (np.exp(Re) - 1)
#Simulate wages
mu_l = 0.015
eps = np.random.normal(0, 0.01, size = (T, M))  #Generate random normal
kappa = 0.15

L = np.ones((T + 1, M))
for t in range(T):
    L[t + 1] = L[t] * np.exp(mu_l + eps[t])

"""
    Compare certainty equivalents
    - There is added value by using the dynamic strategy.
    - CEV of optimal is 1.38 all others are lower< 1.34
"""
gamma = 15
res = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False])
res.plot()

res1 = discrete_grid_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 10, J = 21)
res1.plot()

#No differences if we change kappa / w_initial
#for kappa in [0.1, 0.2 , 0.3]:
#    #(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 50, J = 21, W_init = 0.5).plot()
#    discrete_grid_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 50, J = 21, W_init = 1).plot()



############ WITHOUT INCOME ###############
#Optimal solution:
cev(res.x_opt, Re, Rf, gamma)

# Constant %:
for i in np.arange(0, 1, 0.1):
    cev(np.ones(res.x_opt.shape) * i, Re, Rf, gamma)

############ WITH INCOME ###############
#Simulate wages
mu_l = 0.015
eps = np.random.normal(0, 0.01, size = (T, M))  #Generate random normal
kappa = 0.15

L = np.ones((T + 1, M))
for t in range(T):
    L[t + 1] = L[t] * np.exp(mu_l + eps[t])


#Optimal CEV:
cev1, WT1 = cev_income(res1.x_opt, Re, Rf, kappa, L, gamma) #Appears not optimal!
cev2, WT2 = cev_income(res.x_opt, Re, Rf, kappa, L, gamma) #Random test

(cev1 / cev2 - 1) * 100 #Increase of 0.35% CEV

plt.figure()
plt.hist(WT1, bins = 100)
plt.hist(WT2, bins = 100)
plt.legend(['Income', 'No income'])

avg1 = np.mean(crra(WT1, gamma))
avg2 = np.mean(crra(WT2, gamma))

(avg1 - avg2)/np.abs(avg2) *100 #An 1.4% Increase in the expected utility

WT1_avg = np.mean(WT1)
WT2_avg = np.mean(WT2)
(WT1_avg  - WT2_avg) / WT2_avg * 100 #An 1.4% Increase in the expected terminal wealth

#The volatility is higher:
np.sqrt(np.var(WT1))
np.sqrt(np.var(WT2))


cev(res1.x_opt, Re, Rf, gamma)[0]
cev(res.x_opt, Re, Rf, gamma)[0]

#CEV with constant % in stocks
for i in np.arange(0, 1, 0.1):
    print(cev_income(np.ones(res.x_opt.shape) * i, Re, Rf, kappa, L, gamma)[0])

#CEV with decreasing 1/n
x_n = np.ones(res.x_opt.shape).squeeze() * np.repeat(np.linspace(1, 0, T), M).reshape(T, M)
cev_income(x_n, Re, Rf, kappa, L, gamma)



""" Create Table 1 """
#T = 40# * 4
M = 10000
T_vec = [20, 40, 80, 120]
gamma_vec = [2, 3, 5, 7, 10, 15]

t_len = len(T_vec)
gamma_len= len(gamma_vec)
res_len = 5
VaR_conf = 2.5

res_len = 7 #Also store T and gamma in the array!
colnames = ['T', 'Gamma', 'Mean', 'Vol', 'EU', 'Var', 'CEV']

res = np.zeros((0, res_len)) #empty matrix
for t_idx in range(t_len):
    for g_idx in range(gamma_len):
        T = T_vec[t_idx]
        gamma = gamma_vec[g_idx]

        [Re, Z] = var.simulate(T,  M, seed = 1710)  #Simulate values

        #Set correct dimensions!
        Re = Re.reshape((1, T +1, M))
        Z = Z.reshape(1, T + 1, M)
        Rf = np.full((T + 1, M), rf)  #constant, so everywhere the same

        sol = taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False])
        Cev, WT = cev(sol.x_opt, Re, Rf, gamma) #Appears not optimal!
        mu = np.mean(WT)
        sigma = vol(WT)
        EU = np.mean(crra(WT, gamma))
        VaR = np.percentile(WT, VaR_conf)

        entry = np.array([T, gamma, mu, sigma, EU, VaR, Cev]).reshape(1, res_len)
        res = np.concatenate((res, entry), axis = 0)
res_taylor = res #Store the results

""" """


res = np.zeros((0, res_len)) #empty matrix
for t_idx in range(t_len):
    for g_idx in range(gamma_len):
        T = T_vec[t_idx]
        gamma = gamma_vec[g_idx]

        [Re, Z] = var.simulate(T,  M, seed = 1710)  #Simulate values

        #Set correct dimensions!
        Re = Re.reshape((1, T +1, M))
        Z = Z.reshape(1, T + 1, M)
        Rf = np.full((T + 1, M), rf)  #constant, so everywhere the same

        sol = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False])
        Cev, WT = cev(sol.x_opt, Re, Rf, gamma) #Appears not optimal!
        mu = np.mean(WT)
        sigma = vol(WT)
        EU = np.mean(crra(WT, gamma))
        VaR = np.percentile(WT, VaR_conf)

        entry = np.array([T, gamma, mu, sigma, EU, VaR, Cev]).reshape(1, res_len)
        res = np.concatenate((res, entry), axis = 0)
res_dgrid = res

results = np.concatenate([res_taylor, res_dgrid[:, 2:]], axis = 1)
order = [0, 1, 2, 7, 3, 8, 4, 9, 5, 10, 6, 11]
results = results[:, order] #Sort mu on mu etc


table1_latex = generate_table(results,
                              col1 = [r'\mu', r'\sigma', '\Et[u(W_T)]', r'Var_{0.975}', 'CEV'],
                              col2 = ['Taylor', 'DG'],
                              rownames = ["T", r"\gamma"])

""" Table 2:  Dynamic Income v.s. Static Dynamic"""
M = 10000
T_vec = [20, 40, 80, 120]
gamma_vec = [2, 3, 5, 7, 10, 15]


mu_l = 0.01
sigma_eps = 0.01
kappa = 0.14

res_len = 7 #Also store T and gamma in the array!

res = np.zeros((0, res_len)) #empty matrix
for t_idx in range(t_len):
    for g_idx in range(gamma_len):
        T = T_vec[t_idx]
        gamma = gamma_vec[g_idx]

        [Re, Z] = var.simulate(T,  M, seed = 1710)  #Simulate values

        #Simulate wages
        eps = np.random.normal(0, sigma_eps, size = (T, M))  #Generate random normal
        L = np.ones((T + 1, M))
        for t in range(T):
            L[t + 1] = L[t] * np.exp(mu_l + eps[t])

        #Set correct dimensions!
        Re = Re.reshape((1, T +1, M))
        Z = Z.reshape(1, T + 1, M)
        Rf = np.full((T + 1, M), rf)  #constant, so everywhere the same

        sol = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False])
        Cev, WT = cev_income(sol.x_opt, Re, Rf, kappa, L, gamma) #Appears not optimal!
        mu = np.mean(WT)
        sigma = vol(WT)
        EU = np.mean(crra(WT, gamma))
        VaR = np.percentile(WT, VaR_conf)

        entry = np.array([T, gamma, mu, sigma, EU, VaR, Cev]).reshape(1, res_len)
        res = np.concatenate((res, entry), axis = 0)
res_no_labor = res #Store the results


res = np.zeros((0, res_len)) #empty matrix
for t_idx in range(t_len):
    for g_idx in range(gamma_len):
        T = T_vec[t_idx]
        gamma = gamma_vec[g_idx]

        [Re, Z] = var.simulate(T,  M, seed = 1710)  #Simulate values

        #Simulate wages
        eps = np.random.normal(0, sigma_eps, size = (T, M))  #Generate random normal
        L = np.ones((T + 1, M))
        for t in range(T):
            L[t + 1] = L[t] * np.exp(mu_l + eps[t])

        #Set correct dimensions!
        Re = Re.reshape((1, T +1, M))
        Z = Z.reshape(1, T + 1, M)
        Rf = np.full((T + 1, M), rf)  #constant, so everywhere the same

        sol = discrete_grid_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 50, J = 21)
        Cev, WT = cev_income(sol.x_opt, Re, Rf, kappa, L, gamma) #Appears not optimal!
        mu = np.mean(WT)
        sigma = vol(WT)
        EU = np.mean(crra(WT, gamma))
        VaR = np.percentile(WT, VaR_conf)

        entry = np.array([T, gamma, mu, sigma, EU, VaR, Cev]).reshape(1, res_len)
        res = np.concatenate((res, entry), axis = 0)
res_labor = res #Store the results

results = np.concatenate([res_no_labor, res_labor[:, 2:]], axis = 1)
order = [0, 1, 2, 7, 3, 8, 4, 9, 5, 10, 6, 11]
results = results[:, order] #Sort mu on mu etc

table2_latex = generate_table(results,
                              col1 = [r'\mu', r'\sigma', '\Et[u(W_T)]', r'Var_{0.975}', 'CEV'],
                              col2 = ['No labor', 'Labor'],
                              rownames = ["T", r"\gamma"])


diff = (res_labor[:, 4] - res_no_labor[:, 4])/ np.abs(res_no_labor[:, 4]) * 100

""" With KNW Model """
