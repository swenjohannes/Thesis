# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:42:34 2022

@author: Swen
"""

import pandas as pd
import numpy as np
from KNW import *

"""
    Import the data and make it ready to be used for ML estimation
"""
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

"""
    Bounds and initial value
"""

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
      0.03, 0.03, 0.03, 0.3, #sigma_s
      1, 1, #Lambda0
      1, 1, 1, 1,#Lambda1
      0.05, 0.05, 0.05, 0.05] #sigma_v

x0 = np.array([0.0181, -0.0063, 0.0014, 0.0240, -0.0148, 0.0053, 0.08, -0.19, 0.35,
       0.0002, -0.0001, 0.0061,  0.0452, -0.0053, -0.0076, -0.0211, 0.1659,
       0.403, 0.039, 0.149, -0.381, 0.089, -0.083, 0.01, 0.01, 0.01, 0.01])

maximum_likelihood(pars_0.x, data_knw)

knw = KNW()
res_local = knw.fit(data_knw, lb, ub, pars_0.x,maxiter = 1000)
res_local
maximum_likelihood(res_local.x, data_knw) #8341.57

#Use simulated annealing
res_anneal_0 = knw.fit(data_knw, lb, ub, maxiter = 4000, method = 'anneal')
res_anneal_0

res_anneal = knw.fit(data_knw, lb, ub, res_local.x, maxiter = 1000, method = 'anneal')
res_anneal2 = knw.fit(data_knw, lb, ub, res_anneal_0.x, maxiter = 1000, method = 'anneal')
maximum_likelihood(res_anneal.x, data)

#Simulate from estimated parameters
res = knw.simulate(40, 10000)
res.plot() #Plot series: X1, X2, CPI, S, F0, F5

percentile_plot(res.d_log_Pi, 'Inflation - log returns')
percentile_plot(res.d_log_S, 'Stock - log returns')
percentile_plot(res.d_log_F0, 'Nominal risk-free - log returns')
percentile_plot(res.d_log_F5, '5Y bond funds')


#Check term structure
knw.term_structure()
