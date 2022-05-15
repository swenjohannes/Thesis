# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:00:30 2022

@author: Swen
"""
import numpy as np
import matplotlib.pyplot as plt


from .KNW_pars import KNW_pars
from .fit import anneal, local_search, maximum_likelihood, AN, BN
from .simulation import euler_maruyama, exact_discretization, sim_res

#Miscellaneous
def print_fun(x, f, accepted):
    print("\n Objective: ", f)
    print(KNW_pars(x))

class KNW():
    """
        Creates an KNW object that can be used to fit the parameters to data or
        to run simulations.
    """
    def __init__(self, knw_pars = None, data = None):
        self.pars = knw_pars
        self.data = data

    def fit(self, data, lb, ub, x0 = None,EM = None,
                  maxiter = 1000, method = 'local', display = True):

        """
            Fit the model to the data. By default local search is used. Choose either 'local' or 'anneal'
        """
        obj = lambda x: -maximum_likelihood(x, data)

        if method == 'local':
            res = local_search(obj, x0, maxiter, lb, ub, EM) #use local search
        elif method == 'anneal':
            res = anneal(obj, x0, maxiter, lb, ub, print_fun) #use simulated annealation
        else:
            print("Please select method from: local, anneal")

        self.pars = KNW_pars(res.x) #Transfer into param object!

        #Show the result and return the parameters
        if display:
            print("\nMaximumlikelihood: ", maximum_likelihood(self.pars.x, data))
        return self.pars

    def simulate(self, T = 40, M = 10000, method = 'ED', steps_T = 100, seed = None, discard = 0, h = 1):
        """
            Simulate using the stored parameters. Use either Exact Discretization (ED) or Euler Mayurama approximation (EM). By default ED is used.
        """
        np.random.seed(seed)            #Set the seet for reproducibility

        if method == 'EM':
            Y = euler_maruyama(self.pars, T, M, steps_T)
        elif method == 'ED':
            Y = exact_discretization(self.pars, T, M, h)
        else:
            raise ValueError("Please select method from: Euler Maruyama (EM) or Exact Discretization (ED)")

        initial_values = (self.pars.theta0[2:6] + 1) ** h - 1

        self.res = sim_res(Y, initial_values , discard)
        return self.res

    def term_structure(self, T_max = 30):
        """
            Displays the termstructure coefficients AN and BN.
        """
        #T_max = 30; dt = 0.1
        pars = self.pars
        K = pars.K
        Lambda1 = pars.Lambda1
        Lambda0 = pars.Lambda0
        Sigma_x = pars.Sigma_x
        R0 = pars.R0
        R1 = pars.R1

        tau = np.arange(0.1, T_max, 0.1)

        M = K.T + Lambda1.T @Sigma_x
        A = -AN(M, tau, Lambda0.T @ Sigma_x, R0, R1, delta =0.01)/tau
        B = -BN(M, tau, R1)/tau

        plt.figure()
        plt.plot(tau,  A)
        plt.plot(tau, B.T)
        plt.legend([r'$-A(\tau)/\tau$',
                    r'$-B1(\tau)/\tau$',
                    r'$-B2(\tau)/\tau$'])
        plt.title('Term structure')
        plt.xlabel(r'$\tau$')
        plt.ylabel('y(n)')
