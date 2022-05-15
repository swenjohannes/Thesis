# -*- coding: utf-8 -*-
"""
Created on Sun May  1 22:00:25 2022

@author: Swen
"""
import numpy as np
from numba import njit
from .est_res import est_res

#@timer
#@repeat(10000)
#@njit
def ols(Y, X):
    """
        Fits OLS. Returns an estimation result (est_res) object.
    """
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y #OLS estimator
    res = est_res(Y, X, beta, 'ols')
    return res

ols(np.random.normal(1,2, 10), np.random.normal(1,2, (10, 20)))


#@timer
#@repeat(10000)
@njit
def fast_ols(Y, X, x_new = None, return_beta = False):
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y #OLS estimator
    if not return_beta:
        if x_new is None:
            x_new = X           #Use the new data for y_hat
        ret_val = x_new @ beta  #Return y_hat
    else:
        ret_val = beta          #Return estimated beta's
    return ret_val

#run all options once to compile
fast_ols(np.random.normal(1,2, 10), np.random.normal(1,2, (10, 20)))
fast_ols(np.random.normal(1,2, 10), np.random.normal(1,2, (10, 20)), x_new = np.random.normal(1,2, (10, 20)))
fast_ols(np.random.normal(1,2, 10), np.random.normal(1,2, (10, 20)), x_new = np.random.normal(1,2, (10, 20)), return_beta = True)
fast_ols(np.random.normal(1,2, 10), np.random.normal(1,2, (10, 20)), return_beta = True)




# #Full grid set:
# x_grid_test, G2 =  create_x_grid(100, K) #Construct the discrete grid
# polynomial_basis(x_grid_test, 5)

# theta_basis2 = polynomial_basis(x_grid_test.T, order = [5, 0, False, False, False])

# beta_full = fast_ols(beta, theta_basis, theta_basis2)
# beta_est2 = fast_ols(beta, theta_basis)

# plt.figure()
# plt.plot(x_grid_test, beta_full)
# plt.plot(x_grid, beta_est2)


# plt.figure()
# plt.plot(x_grid_test, beta_full)
# plt.plot(x_grid, beta)

# beta - beta_full

# res = basis @ beta_full.T
# x_opt = x_grid[np.argmax(res, axis = 1)]
# x_opt = x_grid[np.argmax(res, axis = 1)]

# m = 5
# plt.plot(res[m, :])
