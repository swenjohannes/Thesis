# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:50:39 2022

@author: Swen
"""
import numpy as np
from .sim_res import sim_res

def euler_maruyama(knw_pars, T, M, steps_T = 100):
    #steps_T = 100; knw_pars = pars_0
    theta0, theta1, Sigma_y, BN_5 = knw_pars.set_ou_pars() #Get theta0, theta1 etc

    dt = 1 / steps_T     #Interval
    nsteps = int(T / dt) #Total number of steps
    Y = np.zeros((6, T + 1, M))

    eps_h = np.random.normal(0, 1, size = (nsteps, 4, M))
    eps_h[1, :, :]
    Y_t = Y[:, 0]
    #a = theta1 @ Y_t
    for t in range(nsteps):
        #t= 0
        dY_t = (theta0 + theta1 @ Y_t) * dt + np.sqrt(dt) * Sigma_y @ eps_h[t]
        Y_t = Y_t + dY_t

        if (t + 1) % steps_T == 0:
            Y[:, int((t + 1) / steps_T)] = Y_t #store only every T-step (save memory)

    return Y

#sim_res(Y).plot('X1')
### TEST SECTION
#T = 40
#M = 10000
#Y = euler_maruyama(pars_0, T, M, 100)
