# -*- coding: utf-8 -*-
"""
Created on Sun May  1 21:59:43 2022

@author: Swen
"""

import matplotlib.pyplot as plt


class est_res:
    """
        Creates an estimation result object. Has attributes such as X, Y, beta, Y hat and the residuals. Use .plot() to display actual vs fitted.
    """
    def __init__(self, Y, X, beta, method):
        self.X = X
        self.beta = beta
        self.Y = Y
        self.Y_hat = X @ beta
        self.e = self.Y - self.Y_hat
        self.method = method

    def __repr__(self):
        self.plot()
        return("Estimated beta using " + self.method + ": "+ str(self.beta))
    def plot(self):
        plt.figure()
        plt.scatter(self.X[:, 1], self.Y)
        plt.scatter(self.X[:, 1], self.Y_hat)
        plt.title('Actual versus fitted values, used method: ' + self.method)
        plt.legend(['Actual', 'Fitted'])



# @timer
# def func():
#     for _ in range(10000):
#         est_res(Y, X, beta, 'ols')

# func()
