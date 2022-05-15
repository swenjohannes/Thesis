# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:45:56 2022

@author: Swen

This file uses the parameters of the original Draper (2012) paper.

"""
import numpy as np
import matplotlib.pyplot as plt

from KNW import *
from Utils import *
from BGSS import *

"""
    Simulate with the original parameters
"""

x0 = np.array([0.0181, -0.0063, 0.0014, 0.0240, -0.0148, 0.0053, 0.08, -0.19, 0.35,
       0.0002, -0.0001, 0.0061,  0.0452, -0.0053, -0.0076, -0.0211, 0.1659,
       0.403, 0.039, 0.149, -0.381, 0.089, -0.083, 0.01, 0.01, 0.01, 0.01])

pars_0 = KNW_pars(x0)

T = 40
M = 10000

pars_0

knw_0 = KNW(pars_0)
#res = knw_0.simulate(T, M, 'ED')


res = knw_0.simulate(T, M, 'ED', h = 1/4)




"""
    Display the log returns
        -these look good
        -averages look where they should be
        -confidence intervals are ok.
"""
res.plot('S_ret')
res.plot('F0_ret')
res.plot('F5_ret')

#Plot simple returns:
percentile_plot(res.S_ret, title = 'Stock - Simple returns')
percentile_plot(res.F0_ret, title = 'Risk free rate - Simple returns')
percentile_plot(res.F5_ret, title = '5Y bond funds- Simple returns')
percentile_plot(res.Pi_ret, title = 'Inflation - Simple returns')

#Plot log returns
percentile_plot(res.d_log_Pi, title = 'Inflation - log returns')
percentile_plot(res.d_log_S, title = 'Stock - log returns')
percentile_plot(res.d_log_F0, title = 'Nominal risk-free - log returns')
percentile_plot(res.d_log_F5, title ='5Y bond funds', c1 = 2.5 , c2 = 97.5)

knw_0.term_structure()

""" Reproducing - Mumtaz shafiq 'modeling risk premia' '"""

K = pars_0.K
R0 = pars_0.R0
R1 = pars_0.R1
Lambda0 = pars_0.Lambda0
Lambda1 = pars_0.Lambda1
Sigma_x = pars_0.Sigma_x
tau = np.array([0.25, 1, 2, 3, 5 , 10])
n = len(tau)

kernel = K.T + Lambda1.T @Sigma_x
A = AN(kernel, tau, Lambda0.T @ Sigma_x, R0, R1, delta =0.01)
B = BN(kernel, tau, R1)

#Calculate yields
yn = np.zeros((n, T, res.X1.shape[1]))
for t in range(1, T + 1):
    X = np.array([res.X1[t], res.X2[t]])
    yn[:, t - 1] = (-A.reshape(n,1) - B.T @ X)/ tau.reshape(n, 1)

#Calculate path wise volatility
vols = np.sqrt(np.var(yn, axis = 2))

percentile_plot(yn[:, 19], x = tau,
                title = 'Mean bond yield with maturities ranging from 2 to 10 year',
                xlab = 'Maturity', ylab = 'Yield y(n)')

percentile_plot(vols,x = tau,
                title = 'Volatility of bond yield with maturities ranging from 2 to 10 year',
                xlab = 'Maturity', ylab = 'Yield y(n)')

# fig = plt.figure()
# fig.add_subplot(221)   #top left
# percentile_plot(res.d_log_Pi, 'Inflation - log returns')
# fig.add_subplot(222)   #top right
# percentile_plot(res.d_log_S, 'Stock - log returns')
# fig.add_subplot(223)   #bottom left
# percentile_plot(res.d_log_F0, 'Nominal risk-free - log returns')
# fig.add_subplot(224)   #bottom right
# percentile_plot(res.d_log_F5, '5Y bond funds')
# fig.suptitle('Simulation results')
# plt.show()

"""
    Inspect Long-run properties:
"""
long_sim = knw_0.simulate(T = 500, M = 10000, method = 'ED', discard = 50)
percentile_plot(long_sim.F0_ret)

from tabulate import tabulate
rownames = ["Pi", "S", "F0", "F5"]
header = np.array(['Series', 'Sample mean', 'Expected mean'])

#Simple returns
sample_ret = [np.mean(long_sim.Pi_ret),
         np.mean(long_sim.S_ret),
         np.mean(long_sim.F0_ret),
         np.mean(long_sim.F5_ret)]

E_ret = [pars_0.delta_0pi,
                  pars_0.R0 + pars_0.eta_s,
                  pars_0.R0,
                  pars_0.R0 + pars_0.BN_5.T @ pars_0.Sigma_x.T @ pars_0.Lambda0]

data = np.array([rownames, sample_ret, E_ret]).T
data = np.concatenate([header.reshape((1, 3)), data], axis = 0)
tab = tabulate(data, headers='firstrow')
print(tab)

#Long run volatilities
sample_var = [np.var(long_sim.Pi_ret, axis = 0),
              np.var(long_sim.S_ret, axis = 0),
              np.var(long_sim.F0_ret, axis = 0),
              np.var(long_sim.F5_ret, axis = 0)]

sample_var[3] - sample_var[2] #AHA! The variance of bonds - variance of RF is the correct variance

# #Check log-vols
# sample_var = [np.var(long_sim.d_log_Pi, axis = 0),
#               np.var(long_sim.d_log_S, axis = 0),
#               np.var(long_sim.d_log_F0, axis = 0),
#               np.var(long_sim.d_log_F5, axis = 0)]

#Long run volatilities
sample_var = np.mean(sample_var, axis = 1)

E_var = [pars_0.sigma_pi @ pars_0.sigma_pi,
         pars_0.sigma_s @ pars_0.sigma_s,
         0,
         pars_0.BN_5.T @ pars_0.Sigma_x.T @ pars_0.Sigma_x @ pars_0.BN_5
        ]

data = np.array([rownames, sample_var, E_var]).T
data = np.concatenate([header.reshape((1, 3)), data], axis = 0)
tab = tabulate(data, headers='firstrow')
print(tab)

"""
    Compute Re, Rf and Z
"""
T = 40
M = 10000
res = knw_0.simulate(T, M, 'ED', h = 1) #/4)

#T = T - 40
Rf = res.F0_ret
R_S = res.S_ret
R_B = res.F5_ret

R = [R_S, R_B]
Re = R - Rf
Rf = Rf + 1


Inflation =  1/( res.Pi_ret + 1)
Rf = Rf * Inflation
Re = Re * Inflation

#Alternative 1: Results are off!
#Z = np.zeros((2, T + 1, M))
#Z[0] = Rf - 1
#Z[1] = Pi

#for gamma in [5, 15]:
#    discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()
#    taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()


#Alternative 3: use the unobserved states X1 X2:
Z = np.zeros((2, T + 1, M))
Z[0] = res.X1
Z[1] = res.X2

#Alternative 4: use X1, X2, inflation, rf
# Z = np.zeros((4, T + 1, M))
# Z[0] = res.X1
# Z[1] = res.X2
# Z[2] = res.F0_ret
# Z[3] = res.Pi_ret

#Z = np.zeros((2, T + 1, M))
#Z[0] = res.F0_ret
#Z[1] = res.Pi_ret

res = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 10, transform = True).plot()

for gamma in [5, 15]:
    discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50).plot()
    #taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50, method = 'ols').plot()


##CEV:
gamma = 5
#res1 = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50)
res1 = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50)
res1.plot()

#taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50)
#taylor_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 25, method = 'bs')

#res1.x_opt
cev1, WT1 = cev(res1.x_opt, Re, Rf, gamma)
avg1 = np.mean(WT1)
U1 = np.mean(crra(WT1, gamma))
print(cev1)
print(U1)

# Constant %:
test_portfolios = np.array([[0.5, 0.5], [0.6, 0.4], [0.4, 0.6],[0.8, 0.2], [0.2, 0.8]])

for i in test_portfolios:
    #i = test_portfolios[0]
    x_test =np.outer(np.ones(res1.x_opt.shape[1:]).T, i).T.reshape((2, T, M))
    Cev, wt = cev(x_test, Re, Rf, gamma)
    print('avg U: ', np.mean(crra(wt, gamma)))



"""
    With income
"""
#Simulate wages
mu_l = 0.02
eps = np.random.normal(0, 0.02, size = (T, M))  #Generate random normal
kappa = 0.24

L = np.ones((T + 1, M))
for t in range(T):
    L[t + 1] = L[t] * np.exp(mu_l + eps[t])

##CEV:
gamma = 5
W0 = 0

#res2 = discrete_grid_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 50, J = 31, W_init = W0)
res2 = taylor_income(Re, Rf, Z, mu_l, eps, kappa, gamma, basis_order = [2, 0, True, False, False], G = 20, J = 21)
res2.plot()

#res1.x_opt
cev1, WT1 = cev_income(res2.x_opt, Re, Rf, kappa, L, gamma, W_init = W0)
cev2, WT2 = cev_income(res1.x_opt, Re, Rf, kappa, L, gamma, W_init = W0)

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


# a = res.S[1:] / res.S[:-1] - 1
# np.mean(a)
# R_S
# percentile_plot(Re[0])
# plt.plot(Re[0])


"""
    Stock only
"""
Re = Re[0].reshape(1, T + 1, M)

gamma = 5
res1 = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50)
res1.plot()

cev(res1.x_opt, Re, Rf, gamma)[0]
# Constant %:
for i in np.arange(0, 1, 0.1):
    cev(np.ones(res1.x_opt.shape) * i, Re, Rf, gamma)

#Rf.shape
#With constant RF rate:
#rf = 1.02
#Rf = np.full((T + 1, M), 1.02)

res1 = discrete_grid_no_income(Re, Rf, Z, gamma, basis_order = [2, 0, True, False, False], G = 50)
res1.plot()


"""
    Bond only
"""

#Add multivariate version of cev_income
