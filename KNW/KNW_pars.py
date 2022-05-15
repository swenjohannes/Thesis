# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 23:44:57 2022

@author: Swen
"""
#x= x0
import numpy as np
from .fit.ML import expm

class KNW_pars:
    """
        holds all parameters for the KNW model
    """
    def __init__(self, x, ou = True):
        x = np.array(x)
        self.x = x

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

        Lambda0 = x[17:19]
        Lambda1 = x[19:23].reshape(2,2)

        #Other Lambda0/Lambda 1 follow from restriction
        Lambda0 = np.concatenate([Lambda0, np.zeros(2)])
        Lambda1 = np.concatenate([Lambda1, np.zeros((2,2))])

        #Compute the 4th row of Lambda
        Lambda0[3] = (self.eta_s - np.dot(self.sigma_s[0:2], Lambda0[0:2])) \
                        / self.sigma_s[3]
        Lambda1[3] = - np.dot(self.sigma_s[0:2], Lambda1[0:2]) /self.sigma_s[3]

        self.Lambda0 = Lambda0
        self.Lambda1 = Lambda1

        self.sigma_v = np.diag(x[23:27]).astype(float)
        self.Sigma_x = np.concatenate((np.identity(2), np.zeros((2,2))))

        if ou:
            self.set_ou_pars()

    def set_ou_pars(self):
        """
            Derives parameters for the OU process.
        """
        delta_0pi = self.delta_0pi
        delta_1pi = self.delta_1pi
        sigma_pi = self.sigma_pi
        sigma_s = self.sigma_s
        eta_s = self.eta_s
        Sigma_x = self.Sigma_x
        R0 = self.R0
        R1 = self.R1
        Lambda0 = self.Lambda0
        Lambda1 = self.Lambda1
        K = self.K

        M = K.T + Lambda1.T @ Sigma_x
        BN_5 = np.linalg.inv(M) @ (expm(- M * 5) - np.identity(2)) @ R1
        self.BN_5 = BN_5

        self.theta0 = np.array([0,
                  0,
                  delta_0pi - 0.5 * sigma_pi @ sigma_pi,
                  R0 + eta_s - 0.5 * sigma_s @ sigma_s,
                  R0,
                  R0 + BN_5.T @ Sigma_x.T @ Lambda0 - \
                      0.5 * BN_5.T @ Sigma_x.T @ Sigma_x @ BN_5 ]).reshape(6,1)

        theta1 = np.array([-K[0],
                          -K[1],
                          delta_1pi.T,
                          R1.T,
                          R1.T,
                          R1.T + BN_5.T @ Sigma_x.T @ Lambda1 ])

        self.theta1 = np.concatenate((theta1, np.zeros((6,4))), axis =1)
        self.Sigma_y = np.array([Sigma_x.T[0],
                            Sigma_x.T[1],
                            sigma_pi.T,
                            sigma_s.T,
                            np.zeros(4),
                            BN_5.T @ Sigma_x.T])
        return self.theta0, self.theta1, self.Sigma_y, self.BN_5

    def __repr__(self):
        """
            Usage: print(KNW_param object) -> prints the params
        """
        string = "\t pi0: \t"   + str(np.round(self.delta_0pi* 100, 4))   + "%\n"\
                 "\t pi1: \t"   + str(np.round(self.delta_1pi[0]* 100, 4)) + "%\n"\
                 "\t \t \t"     + str(np.round(self.delta_1pi[1]* 100, 4)) + "%\n"\
                 "\t R0: \t"    + str(np.round(self.R0* 100), 4)     + "%\n"\
                 "\t R1: \t"    + str(np.round(self.R1[0])* 100), 4  + "%\n"\
                 "\t \t \t"    + str(np.round(self.R1[1]* 100), 4)   + "%\n"\
                 "\t K: \t"     + str(np.round(self.K[0]), 4)        + "\n"\
                 "\t \t \t"     + str(np.round(self.K[1]), 4)        + "\n"\
                 "\t s_pi: \t"  + str(np.round(self.sigma_pi[0]* 100, 4)) + "%\n"\
                 "\t \t \t"     + str(np.round(self.sigma_pi[1]* 100, 4)) + "%\n"\
                 "\t \t \t"     + str(np.round(self.sigma_pi[2]* 100, 4)) + "%\n"\
                 "\t eta: \t"   + str(np.round(self.eta_s* 100, 4))       + "%\n"\
                 "\t s_s: \t"   + str(np.round(self.sigma_s[0], 4))     + "%\n"\
                 "\t \t \t"     + str(np.round(self.sigma_s[1]* 100, 4)) + "%\n"\
                 "\t \t \t"     + str(np.round(self.sigma_s[2]* 100, 4)) + "%\n"\
                 "\t \t \t"     + str(np.round(self.sigma_s[3]* 100, 4)) + "%\n"\
                 "\t L0: \t"    + str(np.round(self.Lambda0[0:2], 4))     + "\n"\
                 "\t L1: \t"    + str(np.round(self.Lambda1[0], 4))  + "\n"\
                 "\t \t \t"     + str(np.round(self.Lambda1[1], 4))  + "\n"\
                 "\t s_v: \t"   + str(np.round(self.sigma_v.diagonal() * 100, 6))

                 #"\t \t \t"     + str(np.round(self.sigma_pi[3]* 100, 4)) + "%\n"\
                 #"\t \t \t"     + str(np.round(self.Lambda1[2], 4))  + "\n"\
                 #"\t \t \t"     + str(np.round(self.Lambda1[3], 4))  + "\n"\
        string = string.replace(' [', '').replace('[', '').replace(']', '')  #Remove brackets
        return string
