# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:32:15 2022

@author: Swen
"""
import numpy as np
import matplotlib.pyplot as plt

def vol(y):
     return np.sqrt(np.var(y))

class sim_res:
    def __init__(self, Y, initial_values, discard = 0):
        self.Y = Y[:, discard:, :]
        self.X1 = self.Y[0]
        self.X2 = self.Y[1]
        self.log_Pi = self.Y[2]
        self.log_S = self.Y[3]
        self.log_F0 = self.Y[4] #Rename to log_F0
        self.log_F5 = self.Y[5]

        #Log differences: (log returns)
        M = self.Y.shape[2]
        self.d_log_Pi = np.diff(self.log_Pi, axis = 0)
        self.d_log_S = np.diff(self.log_S, axis = 0)
        self.d_log_F0 = np.diff(self.log_F0, axis = 0)
        self.d_log_F5 = np.diff(self.log_F5, axis = 0)

        #Concatenate initial values
        if discard == 0:
            initial_values = np.repeat(initial_values, M ).reshape(4, M)
            self.d_log_Pi = np.concatenate((initial_values[0].reshape(1, M) , self.d_log_Pi), axis = 0)
            self.d_log_S = np.concatenate((initial_values[1, :].reshape(1, M) , self.d_log_S), axis = 0)
            self.d_log_F0 = np.concatenate((initial_values[2, :].reshape(1, M) , self.d_log_F0), axis = 0)
            self.d_log_F5 = np.concatenate((initial_values[3, :].reshape(1, M) , self.d_log_F5), axis = 0)

        #Calculate Simple returns
        self.Pi_ret = np.exp(self.d_log_Pi) - 1
        self.S_ret = np.exp(self.d_log_S) - 1
        self.F0_ret = np.exp(self.d_log_F0) - 1
        self.F5_ret = np.exp(self.d_log_F5) - 1

        #Original series
        self.Pi = np.exp(self.log_Pi)
        self.S = np.exp(self.log_S)
        self.F0 = np.exp(self.log_F0)
        self.F5 = np.exp(self.log_F5)



    def __repr__(self):
        S, F0, F5 = self.S, self.F0, self.F5
        log_S, log_F0, log_F5 = self.log_S, self.log_F0, self.log_F5
        T = len(S) - 1
        #Volatility of 1 period
        string =["\nOne period vol:",
                 vol(S[1]), #stocks
                 vol(F0[1]), #RF interest
                 vol(F5[1]), #5y bond fund
                 "\nT period vol:",
                 vol(S[T]), #stocks
                 vol(F0[T]), #RF interest
                 vol(F5[T]), #5y bond fund
                 "\nT period vol of log:",
                 vol(log_S[T]), #stocks
                 vol(log_F0[T]), #RF interest
                 vol(log_F5[T]), #5y bond fund
                 "\nOne period vol * sqrt(T):",
                 vol(S[1]) * np.sqrt(T), #stocks
                 vol(F0[1]) * np.sqrt(T), #RF interest
                 vol(F5[1]) * np.sqrt(T), #5y bond fund
                 "\n 1 and T period risk-free returns:",
                 np.mean(F0[1]),
                 np.mean(F0[T]),
                 "\nOne period excess returns:",
                 np.mean(S[1]) -np.mean(F0[1]),
                 np.mean(F5[1]) - np.mean(F0[1]),
                 "\nT period excess returns:",
                 np.mean(S[T]) -np.mean(F0[T]),
                 np.mean(F5[T]) - np.mean(F0[T])]

        string = list(map(str, string)) #convert to strings:
        string = '\n'.join(string) #seperate each item on a new line
        return string

    def __plot(self, x, name):
        #Display maximum of 100 lines per plot
        if len(x) > 100:
            x = x[0:100]
        plt.figure()
        plt.plot(x)
        plt.xlabel("Horizon")
        plt.title(name)

    def plot(self, names = None):
        """
            Plot the attributes by calling them by their name!
            example:  sim_res.plot('CPI')
        """
        if names == None:
            names = ['X1', 'X2', 'Pi', 'S', 'F0', 'F5']

        if np.ndim(names) == 0:
            self.__plot(getattr(self, names), names)
        else:
            for name in names:
                self.__plot(getattr(self, name), name)


##### TEST SECTION AND EXAMPLE USAGE:
# sr = sim_res(Y)
# sr.plot()                           #plot all
# sr.plot('CPI')                      #plot ind. serie
# sr.plot(['CPI', 'S', 'F0', 'F5'])   #plot multiple series
