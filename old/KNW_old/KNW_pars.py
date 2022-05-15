# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 23:44:57 2022

@author: Swen
"""
#x= x0
import numpy as np

class KNW_pars:
    """
        holds all parameters for the KNW model
    """
    def __init__(self, x):
        self.delta_0pi = x[0]
        self.delta_1pi = x[[1, 2]]

        self.R0 = x[3]
        self.R1 = x[[4, 5]]


        self.K = np.zeros([2, 2])
        self.K[0, 0] = x[6]
        self.K[1, 0] = x[7]
        self.K[1, 1] = x[8]

        self.sigma_pi = np.concatenate([x[9:12], np.zeros(1)])
        self.eta_s = x[12]
        self.sigma_s = x[13:17]

        self.Lambda0 = x[17:19]
        self.Lambda1 = x[19:23].reshape(2,2)

        self.sigma_v = np.diag(x[23:26]).astype(float)
        self.x = x

    def pars(self):
        """
            Returns all parameters required for simulation!
        """
        return self.delta_0pi, self.delta_1pi, self.R0, self.R1, self.K,\
               self.sigma_pi, self.eta_s, self.sigma_s, self.Lambda0, self.Lambda1

    def __str__(self):
        """
            Usage: print(KNW_param object) -> prints the params
        """
        string = "\t pi0: \t"   + str(self.delta_0pi)   + "\n"\
                 "\t pi1: \t"   + str(self.delta_1pi)   + "\n"\
                 "\t R0: \t"    + str(self.R0)          + "\n"\
                 "\t R1: \t"    + str(self.R1)          + "\n"\
                 "\t K: \t"     + str(self.K[0])        + "\n"\
                 "\t \t \t"     + str(self.K[1])        + "\n"\
                 "\t s_pi: \t"  + str(self.sigma_pi)    + "\n"\
                 "\t eta: \t"   + str(self.eta_s)       + "\n"\
                 "\t s_s: \t"   + str(self.sigma_s)     + "\n"\
                 "\t L0: \t"    + str(self.Lambda0)     + "\n"\
                 "\t L1: \t"    + str(self.Lambda1[0])  + "\n"\
                 "\t \t \t"     + str(self.Lambda1[1])  + "\n"\
                 "\t s_v: \t"   + str(self.sigma_v.diagonal())
        return string



#pars = KNW_pars(x0)
#pars.pars()

#print_x(x0)
#print(pars)
#class KNW():
