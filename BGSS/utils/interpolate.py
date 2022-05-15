# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:26:22 2022

@author: Swen
"""

from numba import njit
import numpy as np

@njit
def interpolate(W, grid, X_opt):
    """
        Interpolates the optimal x* given a point W.
    """
    #grid = W_grid[t]
    #W = Wtp1[0]
    #X_opt = x[:, :, t , m]

    g = len(grid)
    bool_arr = W < grid

    n_true = np.sum(bool_arr) #Number of true's in the array
    if n_true == 1:
        g_max = np.argmax(bool_arr)
        x_g = grid[g_max - 1]
        x_gp1 = grid[g_max]
        y_g = X_opt[g_max - 1]
        y_gp1 = X_opt[g_max]

        x_star = (W - x_g) / (x_gp1 - x_g)
        y_ip = y_g * (1 - x_star) + y_gp1 * x_star
    elif n_true == 0:
        #The value is greater than all grid values.
        #Extrapolate from the last 2 right values
        y_g = X_opt[g - 2]
        y_gp1 = X_opt[g - 1]
        x_g = grid[g - 2]
        x_gp1 = grid[g - 1]

        y_ip = y_g + (W - x_g) * ( y_gp1 - y_g) / (x_gp1 - x_g)

        W - x_g

    else:
        #The value is smaller than all grid values
        #Extrapolate from left
        y_g = X_opt[0]
        y_gp1 = X_opt[1]
        x_g = grid[0]
        x_gp1 = grid[1]

        y_ip = y_g + (W - x_g)  * ( y_gp1 - y_g) / (x_gp1 - x_g)

    if len(X_opt.shape) == 2:
        y_ip = y_ip.reshape(2, 1)

    return y_ip

@njit
def interpolate_mv(W, grid, X_opt):
    """
        The vector input version of interpolate
    """
    #grid = W_grid[t]
    #W = Wtp1[0]
    #X_opt = x[:, :, t , m]
    G, K, T, M = X_opt.shape
    y_ip = np.zeros((K,  T, M))
    for m in range(M):
        y_ip[:, :, m] = interpolate(W[m], grid, X_opt[:, :, :, m])
    return y_ip

def interpolate_x(W, grid, X_opt): #wrapper to make non-negative since Numba doesn't work o.w.
    """
        The user input version of interpolate_mv
    """
    #interpolate_x(Wtp1, W_grid[t], x[:, :, t:T])
    #W = Wtp1; grid = W_grid[t]; X_opt =x[:, :, t:T]
    return interpolate_mv(W, grid, X_opt).clip(min = 0, max = 1)
