# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:21:19 2022

@author: Swen
"""
import numpy as np

def polynomial_basis(z, r = None, order = [2, 0, True, False, False]):
    """"
        Creates a polynomial basis of R and Z with order/covariance options.
    """
    #z = Z[:, t -1]; r = Re[:, t - 1, :]; order = [2, 0, True, False, False]

    #If 1 dim is supplied:
    #Z = np.expand_dims(Z, axis = 0)
    #z = Z[0, t - 1, :].reshape(1,1, M)
    #r = Re[:, t - 1, :]
    if order == None:
        order = [2, 0, True, False, False] #Set default
    #Unpack the items
    z_order = order[0]
    r_order = order[1]
    z_cov = order[2]
    r_cov = order[3]
    rz_cov = order[4]

    M = z.shape[1]
    basis = np.ones(M).reshape(1, M)

    z_len = len(z)
    r_len = len(z)

    for i in range(r_order):
        basis = np.concatenate((basis, r ** (i + 1)))
    for i in range(z_order):
        basis = np.concatenate((basis, z ** (i + 1)))

    #Add z covariants
    if z_cov:
        for j in range(z_len):
            for k in range(j + 1, z_len):
                cov = np.multiply(z[j], z[k]).reshape(1, M)
                basis = np.concatenate((basis, cov))
    #Add r covariants
    if r_cov:
        for j in range(r_len):
            for k in range(j + 1, r_len):
                cov = np.multiply(r[j], r[k]).reshape(1, M)
                basis = np.concatenate((basis, cov))
    #Add r z cross elements
    if rz_cov:
        for j in range(r_len):
            for k in range(z_len):
                cov = np.multiply(r[j], z[k]).reshape(1, M)
                basis = np.concatenate((basis, cov))
    basis = basis.T  #Transpose
    unique, index = np.unique(basis, axis = 1, return_index=True)
    basis = unique[:, index.argsort()] #Prevent multicolinearity!

    #Remove all zero column:
    n_cols = basis.shape[1]
    filter_idx = []
    for i in range(n_cols):
        if basis[:, i].any(): #check if column is all zero's
            filter_idx.extend([i])

    basis = basis[:, filter_idx] #Keep only non-all zero columns
    return basis
