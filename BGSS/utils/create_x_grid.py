# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:18:12 2022

@author: Swen
"""
import numpy as np

def create_x_grid(N, K):
    """
        creates a discrete X grid such that the sum < 1. It also returns the
        number of feasible grid points G
    """
    #N = 5; K = 3;
    if N ** K > 1e6:
        raise RuntimeError('Too many combinations possible. Please select lower N or K!')

    x_grid = np.linspace(0, 1, N + 1).reshape(1, N + 1) #set up 1D 0-> 1 space
    x_rep = np.repeat(x_grid, K, axis = 0).reshape(K, N + 1) #Repeat for K dims

    x_mesh = np.array(np.meshgrid(*x_rep)) #create mesh grid of K dims

    x_grid = x_mesh.T.reshape(-1, K) #Fix dimensions to get gridpoints
    x_grid = x_grid[np.where(np.sum(x_grid, axis = 1) <= 1)] #Remove sum > 1 (infeasible)
    G = len(x_grid) #return nr feasible points
    return x_grid, G
