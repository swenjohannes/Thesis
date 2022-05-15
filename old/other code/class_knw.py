# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 11:52:31 2022

@author: Swen
"""


class knw():
    """
        Creates an object that holds all parameters required for simulation
    """
    def __init__(self, delta_0pi, delta_1pi, R0, R1, K,
                 sigma_pi, eta_s, sigma_s, Lambda0, Lambda1):
        self.delta_0pi = delta_0pi
        self.delta_1pi = delta_1pi
        self.R0 = R0
        self.R1 = R1
        self.K = K
        self.sigma_pi = sigma_pi
        self.eta_s = eta_s
