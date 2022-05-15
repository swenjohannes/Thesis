# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 12:23:09 2022

@author: Swen
"""

import pandas as pd
import numpy as np
from KNW import *


data = pd.read_csv("Data/KNW_data_Netherlands.csv", index_col = 'Date')


annual = np.array(data.iloc[::12, 6])
annual[1:] / annual[:-1]

#Inspection
np.mean(data.Inflation)
np.mean(data['0.25'])




data['LPi'] = np.log(data.Pi)
data['LS'] = np.log(data['^RUI'])
data_knw = data[['0.25', '1.0', '2.0', '3.0', '5.0', '10.0', 'LPi','LS']]
data_knw




lb =[-0.01,#delta_pi0
         -0.02, -0.02, #delta_pi1
         -0.02, #R0
         -0.02, -0.02, #R1
         0, -0.3, 0, #K
         -0.01, -0.01, 0, #sigma_pi
         0.03, #eta_s
         -0.05, -0.05, -0.05, 0.1, #sigma_s
         -1, -1, #Lambda0
         -1, -1, -1, -1,#Lambda1
         1e-6, 1e-6, 1e-6, 1e-6]

ub = [0.03,#delta_pi0
      0.02, 0.02, #delta_pi1
      0.025, #R0
      0.02, 0.02, #R1
      0.5, 0.5, 0.5, #K
      0.01, 0.01, 0.05, #sigma_pi
      0.09, #eta_s
      0.05, 0.05, 0.05, 0.3, #sigma_s
      1, 1, #Lambda0
      1, 1, 1, 1,#Lambda1
      0.1, 0.1, 0.1, 0.1] #sigma_v

x0 = np.array([0.0181, -0.0063, 0.0014, 0.0240, -0.0148, 0.0053, 0.08, -0.19, 0.35,
       0.0002, -0.0001, 0.0061,  0.0452, -0.0053, -0.0076, -0.0211, 0.1659,
       0.403, 0.039, 0.149, -0.381, 0.089, -0.083, 0.01, 0.01, 0.01, 0.01])

pars_0 = KNW_pars(x0)




#Print the inputs to check:
#KNW_pars(lb)
#KNW_pars(ub)
#pars_0

maximum_likelihood(pars_0.x, data_knw)

knw = KNW()
res_local = knw.fit(data_knw, lb, ub, pars_0.x,maxiter = 1000)
res_local   #8341.57

res_anneal_0 = knw.fit(data_knw, lb, ub, maxiter = 4000, method = 'anneal')
res_anneal = knw.fit(data_knw, res_local.x, lb, ub, maxiter = 10000, method = 'anneal')
res_anneal2 = knw.fit(data_knw, res_anneal.x, lb, ub, maxiter = 1000, method = 'anneal')

maximum_likelihood(res_anneal.x, data)

knw.simulate(40, 100000)
knw.res.plot() #Plot series: X1, X2, CPI, S, F0, F5
knw.Y

#knw.fit(data, pars_0.x, lb, ub, maxiter = 100, method = 'anneal') #Doesn't work..

rets = np.exp(np.diff(knw.res.log_S, axis = 0)) - 1
rets_rf = np.exp(np.diff(knw.res.rf, axis = 0)) - 1
rets_inf = np.exp(np.diff(knw.res.inflation, axis = 0)) - 1
rets_f5 = np.exp(np.diff(knw.res.log_B, axis = 0)) - 1

percentile_plot(rets)
percentile_plot(rets_rf)
percentile_plot(rets_inf)
percentile_plot(rets_f5)

#Check A and BN
tau = np.arange(0, 30, 0.1)

M = K.T + Lambda1.T @ Sigma_x
A = -AN(M, tau, Lambda0[0:2], R0, R1, delta =0.01)/tau
B = -BN(M, tau, R1)/tau

plt.figure()
plt.plot(A)
plt.plot(B.T)
plt.legend(['A', 'B1', 'B2'])




def percentile_plot(returns, c1 = 5, c2 = 95):
    #returns = rets; c1 = 5; c2 = 95
    low = np.percentile(returns, c1, axis = 1)
    median = np.percentile(returns, 50, axis = 1)
    mean = np.mean(returns, axis = 1)
    high = np.percentile(returns, c2, axis = 1)

    plot_dat =np.array([low, median, mean, high]).T
    plt.figure()
    plt.plot(plot_dat)



# Simulated longrun means:

# knw_0.simulate(1000, 10000)
# knw_0.res.plot('log_S')
# res = knw_0.res

# avg_cpi =np.mean(np.diff(res.inflation, axis = 0), axis = 0)
# a = np.diff(res.log_S, axis = 0)
# avg_S =np.mean(np.diff(res.log_S, axis = 0), axis = 1)
# plt.plot(avg_S)

# avg_F0 =np.mean(np.diff(res.rf, axis = 0), axis = 0)
# avg_F5 =np.mean(np.diff(res.log_B, axis = 0), axis = 0)


# np.mean(avg_cpi)
# np.mean(avg_S)
# np.mean(avg_F0)
# np.mean(avg_F5)

# theta0,theta1,Sigma_y,BN_5= knw_0.pars.set_ou_pars()

# knw_0.pars.delta_0pi
# knw_0.pars.R0 + knw_0.pars.eta_s
# knw_0.pars.R0
# knw_0.pars.R0 + knw_0.pars.BN_5 @ knw_0.pars.Sigma_x.T @ knw_0.pars.Lambda0

"""
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
T = 20# * 4
M = 10000
gamma = 5

[Re, Z] = var.simulate(T,  M, seed = 1710)  #Simulate values

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

"""  With KNW model !"""
x0 = np.array([0.0181, -0.0063, 0.0014, 0.0240, -0.0148, 0.0053, 0.08, -0.19, 0.35,
       0.0002, -0.0001, 0.0061,  0.0452, -0.0053, -0.0076, -0.0211, 0.1659,
       0.403, 0.039, 0.149, -0.381, 0.089, -0.083, 0.01, 0.01, 0.01, 0.01])

pars_0 = KNW_pars(x0)
T = 20
#T = 1000
M = 10000

pars_0
#With original parameters
knw_0 = KNW(pars_0)
res = knw_0.simulate(T, M, 'EM')

res.plot('X1')

#res.plot(['S', 'F0', 'F5'])
#np.mean(res.S[1:, :] /res.S[:-1, :])
#np.mean(res.F0[1:, :] /res.F0[:-1, :])
#np.mean(res.F5[1:, :] /res.F5[:-1, :])

vol(res.S[19, :] /res.S[18, :])

plt.plot(Rf[:, 1:20])
plt.plot(res.S[1:, 1:50] /res.S[:-1, 1:50])
plt.plot(R_S[:, 1:100])




Rf = np.exp(np.diff(res.rf, axis = 0))
R_S = np.exp(np.diff(res.log_S, axis = 0))
R_B = np.exp(np.diff(res.log_B, axis = 0))
R = [R_S, R_B]
Re = R - Rf
#inf = np.exp(np.diff(res.inflation, axis = 0)) - 1

# plt.plot(R_S)
# R_S.sort()
# R_S[:, 9999]
# plt.plot(Rf)
# plt.plot(R_S)
# plt.plot(res.X2[:, 2:5])
# plt.plot(R_B)
# plt.plot(inf)




# vol(Rf[19])
# vol(res.F0[19])
# vol(res.F5[19])


# (np.mean(res.F0[19]) - 1) ** 1/20 + 1

# a = res.rf
# b = np.diff(res.rf, axis = 0)
# np.mean(np.diff(res.rf, axis = 0), axis = 1)
#a = np.prod(Re + Rf, axis = 1) check -> should be equal to res.F5 or res.S

#Correct for inflation
CPI= res.CPI
Inflation =  CPI[:-1] / CPI[1:]
Rf = np.concatenate((np.repeat(1 + knw_0.pars.R0, M).reshape(1, M), Rf[:-1]))
Rf = Rf * Inflation
Re = Re * Inflation
#Rf = Rf + 1

#pars_0.set_ou_pars()

#np.mean(Re[1, 1:, :], axis = 1)
#Longrun average of S: (should be obtained in a different way)

pars_0.set_ou_pars() #Get theta0, theta1 etc
d, U = np.linalg.eig(pars_0.theta1)  # a = U @ np.diag(D) @ np.linalg.inv(U) #CHECK!
F = np.diag(alpha(d))
mu = U @ F @ np.linalg.inv(U) @ pars_0.theta0
mu = mu.reshape(6, 1)

mu_F0 = mu[4]
mu_S = mu[3] - mu_F0
mu_F5 = mu[5] - mu_F0

print('Stock RE: %.2f%% vol: %.3f%%' % (*mu_S * 100, vol(Re[0, 1]) * 100))
print('Bond RE: %.2f%% vol: %.3f%%' % (*mu_F5 * 100, vol(Re[1, 1]) * 100))

# vol(Re[0, 19]) * 100
# vol(res.S[19])
# vol(res.F0[19])
# vol(res.F5[19])

#np.mean(res.S[1:] / res.S[:-1])


#Compare to empircal:
#avg_F0 = np.mean(np.mean(np.diff(res.rf, axis = 0), axis = 1))
#avg_S =np.mean(np.mean(np.diff(res.log_S, axis = 0), axis = 1)) - avg_F0
#avg_F5 = np.mean(np.mean(np.diff(res.log_B, axis = 0), axis = 1)) -avg_F0

#Alternative 1
#Z = np.zeros((2, T, M))
#Z[0] = Rf - 1
#Z[1] = CPI[:-1]



#Alternative 2: use log scale for CPI
# Z = np.zeros((2, T, M))
# Z[0] = Rf - 1
# inflation = np.diff(res.inflation, axis = 0)
# Z[1] = np.concatenate((np.repeat(knw_0.pars.delta_0pi, M).reshape(1, M), inflation[:-1]))



#Shift TS so t matches index!
mean_Re  = np.repeat(np.array([mu_S, mu_F5]).T, M).reshape(2, 1, M)
Re = np.concatenate((mean_Re, Re), axis = 1)

"""
    Stock & Bond:
        - Stock allocation increases with time and bond allocation decreases.
          Should probably be the other way arround. Explaination can be that
          the same patern is observed if stock and bond are run seperately.
          (See below!!)
        - It appears that the T period volatility of the Risk-Free rate is
          too high.
"""
#Alternative 1: Results are off!
Z = np.zeros((2, T, M))
Z[0] = Rf - 1
Z[1] = CPI[:-1]

for gamma in [5, 15]:
    discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()
    taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()


#Alternative 3: use the unobserved states X1 X2:
Z = np.zeros((2, T + 1, M))
Z[0] = res.X1
Z[1] = res.X2

for gamma in [5, 15]:
    gamma =8
    discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [1, 0, True, False, False], G = 50, method = 'bs').plot()
    taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 25, method = 'bs').plot()


"""
    Try stock only
        - Straight lines as optimal result?
            -> close to Merton's problem if interest rate is assumed to be constant
"""
#With original parameters
knw_0 = KNW(pars_0)
res = knw_0.simulate(T, M, 'ED')

#pars_0
#res.plot('X1')
#res.plot('X2')


#rf = 1.02
#Rf = np.full((T, M), rf)  #constant, so everywhere the same -> problem becomes a Merton problem -> straight line
Rf = np.exp(np.diff(res.rf, axis = 0))
R_S = np.exp(np.diff(res.log_S, axis = 0))
R = [R_S]
Re = R - Rf


Z = np.zeros((2, T + 1, M))
Z[0] = res.X1
Z[1] = res.X2
#Z[2] = res.log_S
#Z[3] = np.concatenate((np.repeat(1 + knw_0.pars.R0, M).reshape(1, M), Rf))
#Shift TS so t matches index!
mean_Re  = np.repeat(np.array([mu_S]).T, M).reshape(1, 1, M)
Re = np.concatenate((mean_Re, Re), axis = 1)


gamma = 6
#discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [1, 0, False, False, False], G = 50).plot()
#discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 100).plot()
res = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 1, True, False, False], G = 100)
res.plot()
np.mean(res.x_mean)

#taylor_no_income(Re, Rf, Z, gamma, basis_order = [1, 0, False, False, False]).plot()
taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False]).plot()
taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 1, False, False, False]).plot()

T = 40
for gamma in [2, 5, 7, 10]:
    taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False]).plot()

"""
    Try bond only
        - No straight lines
        - Bond allocation decreases with time
        - The decreasing function is probably because x1 x2 start at 0 and develope over time

"""
T = 20
#With original parameters
knw_0 = KNW(pars_0)
res = knw_0.simulate(T, M)

#a = np.exp(np.diff(res.log_B, axis = 0)) - Rf
rf = 1.024
Rf = np.full((T, M), rf)  #constant, so everywhere the same

#Rf = np.exp(np.diff(res.rf, axis = 0))
R_F5 = np.exp(np.diff(res.log_B, axis = 0))
R = [R_F5]
Re = R - Rf

pars_0.set_ou_pars() #Get theta0, theta1 etc
d, U = np.linalg.eig(pars_0.theta1)  # a = U @ np.diag(D) @ np.linalg.inv(U) #CHECK!
F = np.diag(alpha(d))
mu = U @ F @ np.linalg.inv(U) @ pars_0.theta0
mu = mu.reshape(6, 1)

mu_F0 = mu[4]
mu_S = mu[5] - mu_F0

print('Bond RE: %.2f%% vol: %.3f%%' % (*mu_F5 * 100, vol(Re[0, 1]) * 100))

Z = np.zeros((2, T + 1, M))
Z[0] = res.X1
Z[1] = res.X2

#Shift TS so t matches index!
mean_Re  = np.repeat(np.array([mu_S]).T, M).reshape(1, 1, M)
Re = np.concatenate((mean_Re, Re), axis = 1)


gamma = 15
#discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [1, 0, False, False, False], G = 50).plot()
discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()
#taylor_no_income(Re, Rf, Z, gamma, basis_order = [1, 0, False, False, False]).plot()
taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False]).plot()


T = 40
for gamma in [2, 5, 7, 10, 15]:
    discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()
    #taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False]).plot()



"""  NOW FIND THE OPTIMAL ASSET ALLOCATIONS"""

for gamma in [2, 5, 7, 10]:
    x_opt = taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False])
    stack_plot(x_opt, "No income, taylor grid, risk aversion: " + str(gamma))

for gamma in [2, 5, 7, 10]:
    x_opt = discrete_grid_no_income(Re, Rf, Z, gamma, G = 50, basis_order = [2, 1, True, True, True])
    #x_opt = discrete_grid_no_income(Re, Rf, Z, gamma, G = 50, basis_order = [2, 0, True, False, False])
    stack_plot(x_opt, "No income, discrete grid, risk aversion: " + str(gamma))

    # x_opt = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 1, True, True, True])
    # stack_plot(x_opt, "No income, discrete grid, risk aversion: " + str(gamma))


    #x_opt = discrete_grid_income(Re, Rf, Z, mu_l, eps, 0.00001, gamma, basis_order = [2, 1, True, True, True])
    #stack_plot(x_opt, "With income, discrete grid, risk aversion: " + str(gamma))


mean_no_income = np.mean(x_opt[:, :], axis = 2)
np.mean(x_opt[:, T -1], axis = 1)

"""
    With income
"""
#knw_0.pars.set_ou_pars()
#knw_0.pars.theta1

#Simulate wages
mu_l = 0.02
eps = np.random.normal(0, 0.0, size = (T, M))  #Generate random normal

L = np.ones((T + 1, M))
for t in range(T):
     L[t + 1] = L[t] * np.exp(mu_l + eps[t])
plt.figure()
plt.plot(L)




mu_l = 0.01
for gamma in [2, 5, 7, 12]:
    x_opt = discrete_grid_income(Re, Rf, Z, mu_l, eps, 0.12, gamma, basis_order = [2, 0, True, False, False])
    stack_plot(x_opt, "With income, discrete grid, risk aversion:" + str(gamma))

    x_opt =taylor_income(Re, Rf, Z, mu_l, eps, 0.12, gamma, basis_order = [2, 1, True, True, True])
    stack_plot(x_opt, "With income, Taylor approx, risk aversion:" + str(gamma))





mean_income = np.mean(x_opt[:, :], axis = 2)
