# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:07:26 2022

@author: Swen
"""
import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import minimize, LinearConstraint, Bounds
#from Timer import timer
from numba import njit

""" FIT EXAMPLE """

""" Using Draper 2014 parameters"""
## PARAMETERS
delta_0pi = 0.0181
delta_1pi = np.array([-0.0063, 0.0014])

R0 = 0.0240
R1 = np.array([-0.0148, 0.0053])

K = np.array([[0.08, 0],
              [-0.19, 0.35]])

sigma_pi = np.array([0.0002, -0.0001, 0.0061, 0])

eta_s = 0.0452
sigma_s = np.array([-0.0053, -0.0076, -0.0211, 0.1659])


Lambda0 = np.array([0.403, 0.039, 0, 0])
Lambda1 = np.array([[0.149, -0.381], [0.089, -0.083], [0, 0], [0, 0]])


""" Using 1972.4 - 2013 """
# delta_0pi = 0.0187
# delta_1pi = np.array([0.0048, 0.0043])

# R0 = 0.0253
# R1 = np.array([0.0129, 0.0091])

# K = np.array([[0.35, 0],
#               [-0.20, 0.08]])

# sigma_pi = np.array([-0.0002, -0.0002, 0.0061, 0])

# eta_s = 0.0454
# sigma_s = np.array([-0.0032, 0.0088, -0.0209, 0.166])


# Lambda0 = np.array([-0.200, -0.347, 0, 0])
# Lambda1 = np.array([[0.135, -0.080], [0.401, -0.068], [0, 0], [0, 0]])



""" Calculate other parameters """
Lambda0 = np.concatenate([Lambda0, np.zeros(2)])
Lambda1 = np.concatenate([Lambda1, np.zeros((2,2))])


#Compute the 4th row of Lambda
Lambda0[3] = (eta_s - np.dot(sigma_s[0:2], Lambda0[0:2])) / sigma_s[3]
Lambda1[3] = - np.dot(sigma_s[0:2], Lambda1[0:2]) /sigma_s[3]

Sigma_x = np.concatenate((np.identity(2), np.zeros((2,2))))


#Term structure
M = (K.T + Lambda1.T @ Sigma_x) # Pricing kernel

@njit
def expm(A):
    d, U = np.linalg.eig(A)
    return U @ np.diag(np.exp(d)) @ np.linalg.inv(U)

#Run once to compile!
expm(-M)



#BN_0 = np.linalg.inv(M) @ (expm(- M * 0) - np.identity(2)) @ R1
#BN_1 = np.linalg.inv(M) @ (expm(- M * 1) - np.identity(2)) @ R1
BN_5 = np.linalg.inv(M) @ (expm(- M * 5) - np.identity(2)) @ R1
#BN_10 = np.linalg.inv(M) @ (expm(-M * 10) - np.identity(2)) @ R1

theta0 = np.array([0,
          0,
          delta_0pi - 0.5 * sigma_pi @ sigma_pi,
          R0 + eta_s - 0.5 * sigma_s @ sigma_s,
          R0,
          R0 + BN_5.T @ Sigma_x.T @ Lambda0 - \
              0.5 * BN_5.T @ Sigma_x.T @ Sigma_x @ BN_5 ]).reshape(6,1)

#a = BN_5.T
theta1 = np.array([-K[0],
                  -K[1],
                  delta_1pi.T,
                  R1.T,
                  R1.T,
                  R1.T + BN_5.T @ Sigma_x.T @ Lambda1 ])

theta1 = np.concatenate((theta1, np.zeros((6,4))), axis =1)

Sigma_y = np.array([Sigma_x.T[0],
                    Sigma_x.T[1],
                    sigma_pi.T,
                    sigma_s.T,
                    np.zeros(4),
                    BN_5.T @ Sigma_x.T])

""" EXACT DISCRETIZATION """
#Eigenvalue decomposition for theta 1:
d, U = np.linalg.eig(theta1)  # a = U @ np.diag(D) @ np.linalg.inv(U) #CHECK!
U_inv = np.linalg.inv(U)

#a = U @ np.diag(d) @ np.linalg.inv(U)

def alpha(X):
    return [(np.exp(x) - 1) / x if x != 0 else 1 for x in X]


T = 40 #Simulation periods
M = 10000
Y = np.zeros((6, T + 1, M))

for h in range(1, T + 1):
    #h =1
    #Compute required parts
    gamma_h = U @ np.diag(np.exp(d * h )) @ U_inv
    F = h * np.diag( alpha(d * h))
    mu_h = U @ F @ U_inv @ theta0

    inner = np.add.outer(d, d) * h
    V = U_inv @ Sigma_y @ Sigma_y.T @ U_inv.T * h *[alpha(v) for v in inner]
    sigma_h = U @ V @ U.T

    #b = U_inv @ Sigma_y @ Sigma_y.T @ U_inv.T
    #c =  U_inv @ Sigma_y @ Sigma_y.T
    #a= h * np.array([alpha(v) for v in inner])

    eps_h = np.random.multivariate_normal(np.zeros(6), sigma_h, M)
    #a =  mu_h.reshape(6, 1) + gamma_h @ Y[h] + eps_h.T

    Y[:, h] = mu_h.reshape(6, 1) + gamma_h @ Y[:, 0] + eps_h.T

#from numba import njit

#@njit
def simulate_EM(T, M, theta0, theta1, Sigma_y, steps_T = 5):

    delta = 1 / steps_T     #Interval
    nsteps = int(T / delta) #Total number of steps

    #Y = np.zeros((T + 1, 6, M))
    Y = np.zeros((6, T + 1, M))

    eps_h = np.random.normal(size = (nsteps, 4, M))

    Y_t = Y[:, 0]
    for t in range(nsteps):
        Y_t = Y_t + theta0 * delta + theta1 @ Y_t * delta + \
                    np.sqrt(delta) * Sigma_y @ eps_h[t]
        if (t + 1) % steps_T == 0:
            Y[:, int((t + 1) / steps_T)] = Y_t #store only every T-step (save memory)

    return Y

T = 40
M = 10000
Y = simulate_EM(T, M, theta0, theta1, Sigma_y, 100)


#Volatility of stocks:
np.sqrt(np.var(Y[3, 1, :]))
theta0[3] - theta0[4]

#Volatility of RF rate:
np.sqrt(np.var(Y[4, 1, :]))
np.mean(Y[4, 1, :])
np.mean(Y[4, T, :])


#Volatility of bond fund:
np.sqrt(np.var(Y[5, 1, :]))

#Excess return on bond -> - 1.5 % a year
theta0[5] - theta0[4]
np.mean(Y[5, 1, :]) -np.mean(Y[4, 1, :])
np.mean(Y[5, T, :]) - np.mean(Y[4, T, :])


np.sqrt(np.var(Y[3, 1, :]))
theta0[3] - theta0[4]

np.mean(Y[5], axis =1) - np.mean(Y[4], axis =1)
# a = np.exp(Y[Series['lF0']])
# a = np.exp(Y[Series['lF5']])
# a = np.exp(Y[Series['lS']])
# np.mean(a[40])
# np.cov(a[40])

"""
    Determine RE and Z
"""
from enum import IntEnum
class Series(IntEnum):
    X1 = 0
    X2 = 1
    lPi = 2
    lS = 3
    lF0 = 4
    lF5 = 5

#Determine Returns from log returns
Rf = np.exp(np.diff(Y[Series['lF0']], axis = 0)) - 1
R = np.exp(np.diff(Y[[Series['lS'], Series['lF5']]], axis = 1)) -1
Re = R - Rf

#Correct for inflation
Pi = np.exp(Y[Series['lPi']])
Inflation = Pi[:-1] / Pi[1:]
Rf = np.concatenate((np.repeat(R0, M).reshape(1, M), Rf[:-1]))
Rf = Rf * Inflation
Re = Re * Inflation
Rf = Rf + 1

#Shift TS so t matches index!
Re = np.concatenate((np.ones((M, 2)).reshape(2, 1, M), Re), axis = 1)


#Alternative 1
Z = np.zeros((2, T, M))
Z[0] = Rf - 1
Z[1] = Pi[:-1]

# #Alternative 2
# Z = np.zeros((2, T, M))
# rf =  np.exp(np.diff(Y[Series['lF0']], axis = 0)) - 1
# rf = np.concatenate((np.repeat(R0, M).reshape(1, M), rf[:-1]))
# Z[0] = rf
# Pi =  np.exp(np.diff(Y[Series['lPi']], axis = 0)) - 1
# Pi =  np.concatenate((np.repeat(delta_0pi, M).reshape(1, M), Pi[:-1]))
# Z[1] = Pi

#Alternative 3
#Z = np.zeros((1, T, M))
#Z[0] = Rf - 1

gamma = 5


""" DISPLAY RESULTS"""

plt.figure()
plt.plot(Y[Series['X1']])
plt.xlabel("Horizon")
plt.title("X1")

plt.figure()
plt.plot(Y[Series['X2']])
plt.xlabel("Horizon")
plt.title("X2")

plt.figure()
plt.plot(np.exp(Y[Series['lPi']]))
plt.xlabel("Horizon")
plt.ylabel("CPI - index")
plt.title("Inflation index")

plt.figure()
plt.plot(np.exp(Y[Series['lS']]))
plt.xlabel("Horizon")
plt.ylabel("Index")
plt.title("Stock development")

plt.figure()
plt.plot(np.exp(Y[Series['lF0']]))
plt.xlabel("Horizon")
plt.ylabel("Index")
plt.title("Cash development (Nominal terms)")

plt.figure()
plt.plot(np.exp(Y[Series['lF5']]))
plt.xlabel("Horizon")
plt.ylabel("Index")
plt.title("5y bond fund development (Nominal terms)")



"""
    Solve MV asset allocation problem using discrete grid:
"""
def create_x_grid(N):
    x_grid = np.linspace(0, 1, N + 1)
    #Find all combinations
    x_grid = np.array(np.meshgrid(x_grid, x_grid)).T.reshape(-1, 2)
    #Remove sum > 1 (infeasible)
    x_grid = x_grid[np.where(np.sum(x_grid, axis = 1) <= 1)]
    G = len(x_grid)
    return x_grid, G

def stack_plot(x_opt, T, title):
    x_ax = list(range(T))
    y = np.mean(x_opt, axis = 2).T
    y_stack = np.cumsum(y, axis = 1)
    y_stack = np.concatenate((np.zeros(T).reshape(T, 1), y_stack), axis =1)
    y_stack = np.concatenate((y_stack, np.ones(T).reshape(T, 1)), axis = 1)
    y_stack = y_stack.T

    plt.figure()
    for i in range(len(y_stack)-1):
        plt.fill_between(x_ax, y_stack[i], y_stack[i + 1])
    plt.title(title)
    plt.xlabel("Horizon")
    plt.ylabel("Asset weight")
    plt.legend(['Stock', 'Bonds', 'Cash'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def polynomial_basis2(z, r = None, z_order = 2, r_order=0,
                      z_cov = False, r_cov = False, rz_cov = False):
    #Z = np.expand_dims(Z, axis = 0)
    #z = Z[0, t - 1, :].reshape(1,1, M)
    #r = Re[:, t - 1, :]

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
    return basis.T


def cond_ex(Y, X, display = False):
        #X, Y  = basis,  psi[t] * Re[t]         #Testing purpose
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y #OLS estimator
        Y_hat = X @ beta                        #Return fitted values

        #Show fitted values versus estimated!
        if display == True:
            plt.figure()
            plt.scatter(X[:, 1], Y)
            plt.scatter(X[:, 1], Y_hat, c='red')
        return Y_hat        #Return predicted values and the estimator


gamma = 7
#Construct the discrete grid
x_grid, G = create_x_grid(5)

psi = np.ones((T + 1, M)) / (1 - gamma) #First psi = 1
x = np.zeros((2, T, M))     #Store opt values

#Solve T -> 2
for t in reversed(range(2, T + 1)):
    #t = T #t =2
    print(t)
    basis = polynomial_basis2(Z[:, t - 1, :], Re[:, t - 1, :],
                      z_order = 2, r_order = 2, z_cov = True, r_cov = True, rz_cov = True)

    psi_g = np.zeros((G, M))
    for g in range(G):
        psi_g[g] = cond_ex((np.dot(x_grid[g], Re[:, t]) + Rf[t - 1]) ** \
                           (1 - gamma) * psi[t], basis) # , True)

    x[:, t - 1] = x_grid[np.argmax(psi_g, axis = 0)].T
    psi[t - 1] = np.max(psi_g, axis = 0)

#Solve last period (only 1 point, so no conditional expectations)
t = 1
print(t)
psi_g = np.zeros((G, M))
for g in range(G):
    psi_g[g] =(np.dot(x_grid[g], Re[:, t]) + Rf[t - 1]) ** (1 - gamma) * psi[t]
x[:, t - 1] = x_grid[np.argmax(psi_g, axis = 0)].T
psi[t - 1] = np.max(psi_g, axis = 0)

stack_plot(x, T, "No income, discrete grid")


"""
    Solve MV asset allocation problem using Taylor approximation
    - Using minimize routine to add constraints. (very slow)
    - Difficulty with RF(t)
"""
########### NOT RUN: TOO SLOW ########################
# K = 2
# psi = np.ones((T + 1, M)) # / (1 - gamma) #First psi = 1
# x = np.zeros((K, T, M))     #Store opt values

# con = LinearConstraint(np.ones(K), 0, 1)        #Sum < 1
# bounds = Bounds(np.zeros(K), np.ones(K))        #all between 0, 1
# obj = lambda y, a, b: -1 * (y @ a - 0.5 * gamma* y @ b @ y)
# x0 = np.zeros(K)

# for t in reversed(range(1, T + 1)):
#     print(t)
#     #t = T
#     basis = polynomial_basis2(Z[:, t - 1, :], Re[:, t - 1, :],
#                       z_order = 2, r_order = 0, z_cov = True)
#     if t == 1:
#         basis = basis[:, 1:4] #Prevent multi-colinearity

#   #SOLVE SECOND ORDER APPROXIMATION
#     a_hat = np.zeros((M, K)) #Asset wise conditinal expectation
#     for i in range(K):
#         a_hat[:, i] = cond_ex(Rf[t - 1] ** -gamma * Re[i, t] * psi[t], basis)

#     b_hat = np.zeros((M, K, K))
#     for i in range(K):
#         for j in range(K):
#             b_hat[:, i, j] = cond_ex(Rf[t - 1] ** (-1 -gamma)  * Re[i, t, :] * Re[j, t, :] * psi[t],  basis)

#     for m in range(M):
#         sol = minimize(obj,  x0,
#                    args = (a_hat[m], b_hat[m]),
#                    constraints= con, bounds = bounds)
#         x[:, t - 1, m] = sol.x

#     psi[t - 1] = psi[t] * (np.sum(x[:, t- 1] *  Re[:, t], axis = 0) + Rf[t -1]) ** (1 - gamma)

# stack_plot(x, T, "No income - 2nd order Taylor approx + minimize")

"""
    Solve MV asset allocation problem using Taylor approximation and discrete grid

"""
#Construct the discrete grid
x_grid, G = create_x_grid(10)

K = 2
psi = np.ones((T + 1, M)) #First psi = 1
x = np.zeros((K, T, M))     #Store opt values
for t in reversed(range(1, T + 1)):
    print(t)
    #t = T
    basis = polynomial_basis2(Z[:, t - 1, :], Re[:, t - 1, :],
                      z_order = 2, r_order = 0, z_cov = True)
    if t == 1:
        basis = basis[:, 1:4] #Prevent multi-colinearity

  #SOLVE SECOND ORDER APPROXIMATION
    a_hat = np.zeros((M, K)) #Asset wise conditinal expectation
    for i in range(K):
        a_hat[:, i] = cond_ex(Rf[t - 1] ** -gamma * Re[i, t] * psi[t], basis)

    b_hat = np.zeros((M, K, K))
    for i in range(K):
        for j in range(K):
            b_hat[:, i, j] = cond_ex(Rf[t - 1] ** (-1 -gamma)  * Re[i, t, :] * Re[j, t, :] * psi[t],  basis)

    y_g = np.zeros((G, M))
    for g in range(G):
        y_g[g] = x_grid[g] @ a_hat.T - 0.5 * gamma * x_grid[g] @ b_hat @ x_grid[g].T

    x[:, t - 1] = x_grid[np.argmax(y_g, axis = 0)].T
    psi[t - 1] = psi[t] * (np.sum(x[:, t- 1] *  Re[:, t], axis = 0) + Rf[t -1]) ** (1 - gamma)

stack_plot(x, T, "No income - 2nd order Taylor + discrete grid")

##### SOLVE LAST PERIOD

#TO DO!!!!!!!!!!

"""
    Required functions for income problem
"""

@njit
def interpolate(W, grid, X_opt):
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
    #grid = W_grid[t]
    #W = Wtp1[0]
    #X_opt = x[:, :, t , m]
    M = len(W)
    T = X_opt.shape[2]
    y_ip = np.zeros((2,  T, M))
    for m in range(M):
        y_ip[:, :, m] = interpolate(W[m], grid, X_opt[:, :, :, m])
    return y_ip

def interpolate_x(W, grid, X_opt): #wrapper to make non-negative since Numba doesn't work o.w.
    return interpolate_mv(W, grid, X_opt).clip(min = 0, max = 1)

def psi2(x_hat, Re, Rf):
    Ret = np.sum(x_hat *  Re, axis = 0) + Rf
    rev_rp = np.flip(Ret, axis = 0)  #Flip returns
    psi_t = np.cumprod(rev_rp, axis = 0)    #Compute cumprod from t to T
    psi_t = np.flip(psi_t, axis = 0)         #Flip back
    return psi_t

def annuity(r, N, A):
    c = (1 + r) ** N - 1
    return A * ( c/r + c)


def chi(psi_hat, eps, mu_l):
    #
    E = np.exp(np.cumsum(eps + mu_l, axis = 0))
    chi = np.sum(E * psi_hat, axis = 0)
    return chi

"""
    Solve MV asset allocation with income - Discrete grid

    - Problems with wealth grid interpolation
"""

#Simulate wages
mu_l = 0.02
eps = np.random.normal(0, 0.05, size = (T, M))  #Generate random normal

L = np.ones((T + 1, M))
for t in range(1, T + 1):
    print(t)
    L[t] = L[t - 1] * np.exp(mu_l + eps[t - 1])

plt.plot(L)

kappa = 0.12

rf = 1.024
gamma =  5                       #Risk aversion parameter

# J = 11
# contribution = annuity(rf - 1,  np.linspace(1, T, T), kappa)
# minmax = np.exp(np.linspace(-3, 3, J))
# W_grid = np.outer(contribution, minmax)

#Construct W grid
J = 11
d = 0.20
minmax = np.linspace(1 - d, 1 + d, J)*rf - 1
W_grid = np.zeros((T, J))
for j in range(J):
    W_grid[:, j] = annuity(minmax[j] , np.linspace(1, T, T), kappa)

#Contruct X grid
x_grid, G = create_x_grid(5)

K = 2
x = np.zeros((J, K, T, M))          #Store opt values

###### SOLVE FIRST PERIOD #################
t = T
basis = polynomial_basis2(Z[:, t - 1, :], Re[:, t - 1, :],
                   z_order = 2, r_order = 0, z_cov = True)
for j in range(J):
    W_tp1 = W_grid[t - 1, j] + kappa
    psi_g = np.zeros((G, M))
    for g in range(G):
        psi_g[g] = cond_ex((W_tp1 * (np.dot(x_grid[g], Re[:, t]) + Rf[t -1])) \
                        ** (1-gamma) / (1 - gamma)  , basis)

    x[j, :, t - 1] = x_grid[np.argmax(psi_g, axis = 0)].T

###### SOLVE T -1 -> t = t + 1  #################
for t in reversed(range(2, T)):
    #t = T -1
    print(t)
    basis = polynomial_basis2(Z[:, t - 1, :], Re[:, t - 1, :],
                       z_order = 2, r_order = 0, z_cov = True)
    for j in range(J):
        #j = 0
        psi_g = np.zeros((G, M))
        x_hat = np.zeros((G, K, T - t, M))
        for g in range(G):
            #g = 0
            Wtp1 = (W_grid[t - 1, j] + kappa) * (np.dot(x_grid[g], Re[:, t]) + Rf[t -1])

            a = x_hat[1]
            b = psi_hat = psi2(x_hat[g], Re[:, t:T], Rf[t:T])           #Compute phi_hat from x_hat

            x_hat[g] = interpolate_x(Wtp1, W_grid[t], x[:, :, t:T]) #Interpolate x* from Wtp1
            psi_hat = psi2(x_hat[g], Re[:, t:T], Rf[t:T])           #Compute phi_hat from x_hat
            chi_hat = chi(psi_hat, eps[t:T], mu_l)               #Compute chi_hat from x_hat

            #Compute expected value for gridpoint
            W_hat = Wtp1 * psi_hat[0] + kappa * chi_hat
            psi_g[g] = cond_ex(W_hat.squeeze() ** (1 - gamma) / (1 - gamma), basis) #, True)

        g_idx = np.argmax(psi_g, axis = 0)
        x[j, :, t - 1] = x_grid[g_idx].T
        for m in range(M):
            x[j, :, t:T, m] = x_hat[g_idx[m], :, : , m] #Store X_hat that was used!
    #stack_plot(x[5, :, (t-1):T, :], T - t, "Income - Discrete grid")   #Display the results
###### SOLVE LAST PERIOD  #################
t = 1
print(t)
J_g = np.zeros((G, M))
x_hat = np.zeros((G, K, T - t, M))
for g in range(G):
    Wtp1 = kappa * (np.dot(x_grid[g], Re[:, t]) + Rf[t -1])

    x_hat[g] = interpolate_x(Wtp1, W_grid[t], x[:, :, t:T]) #Interpolate x* from Wtp1
    psi_hat = psi2(x_hat[g], Re[:, t:T], Rf[t:T])           #Compute phi_hat from x_hat
    chi_hat = chi(psi_hat, eps[t:T], mu_l)               #Compute chi_hat from x_hat

    W_hat = Wtp1 * psi_hat[0] + kappa * chi_hat
    J_g[g] = W_hat ** (1 - gamma) / (1 - gamma)

#DETERMINE FINAL SOLUTION:
g_idx = np.argmax(J_g, axis = 0)
x_opt = np.zeros((K, T, M))
x_opt[:, 0, :] = x_grid[g_idx].reshape(K, M)

for m in range(M):
    x_opt[:, 1:, m] = x_hat[g_idx[m], :, :, m]


stack_plot(x_opt , T, "Income - Discrete grid")   #Display the results


"""
    SOLVE USING TAYLOR APPROX (& discrete grid to solve constrained problem)
"""

#Solve period T
t = T
basis = polynomial_basis2(Z[:, t - 1, :], Re[:, t - 1, :],
                   z_order = 2, r_order = 0, z_cov = True)
x = np.zeros((J, K, T, M))          #Store opt values
x_grid, G = create_x_grid(5)

for j in range(J):
    W_hat = (W_grid[t - 1, j] + kappa) * Rf[t - 1]

    a_hat = np.zeros((M, K)) #Asset wise conditinal expectation
    for k in range(K):
        a_hat[:, k] = cond_ex(W_hat ** -gamma * Re[k, t], basis)

    b_hat = np.zeros((M, K, K))
    for p in range(K):
        for q in range(K):
             b_hat[:, p, q] = cond_ex(W_hat ** (-gamma - 1) * Re[p, t, :] * Re[q, t, :], basis)

    y_g = np.zeros((G, M))
    for g in range(G):
         #g = 8
         y_g[g] = x_grid[g] @ a_hat.T - \
             0.5 * gamma * x_grid[g] @ b_hat @ x_grid[g].T * (W_grid[t - 1, j] + kappa)
    x[j, :, t - 1] = x_grid[np.argmax(y_g, axis = 0)].T

    # ALTERNATIVE:
    # for m in range(M):
    #     sol = minimize(obj,  x0,
    #                    args = (a_hat[m], b_hat[m]),
    #                    constraints= con, bounds = bounds)
    #     x[j, :, t - 1, m] = sol.x

# np.mean(x[:, 0, t -1])
# np.mean(x[:, 1, t -1])

# a = x[:, :, t]
# a = np.mean(x[:, :, t], axis = 2)
# plt.plot(a)

# a = x[:, :, t - 1]
# a = np.mean(x[:, :, t - 1], axis = 2)
# plt.plot(a)

for t in reversed(range(1, T)):
    #t = T -1 #t = t -1
    print(t)
    basis = polynomial_basis2(Z[:, t - 1, :], Re[:, t - 1, :],
                       z_order = 2, r_order = 0, z_cov = True)

    if t == 1:
        basis = basis[:, 1:4] #Prevent multi-colinearity

    for j in range(J):
        #j = 0
        Wtp1 = (W_grid[t - 1, j] + kappa) * Rf[t - 1]

        #Interpolate Wtp1 so we obtain the optimal sequence x_star
        x_hat  = interpolate_x(Wtp1, W_grid[t], x[:, :, t:T, :])
        psi_hat = psi2(x_hat, Re[:, t:T], Rf[t:T]) #Compute phi_hat from x_hat
        chi_hat = chi(psi_hat, eps[t:T], mu_l) #Compute chi_hat from x_hat

        W_hat = Wtp1 * psi_hat[0] + kappa * chi_hat #Compute terminal wealth

        a_hat = np.zeros((M, K)) #Asset wise conditinal expectation
        for i in range(K):
            a_hat[:, i] = cond_ex(W_hat.squeeze() ** -gamma * Re[i, t], basis)

        b_hat = np.zeros((M, K, K))
        for p in range(K):
            for q in range(K):
                b_hat[:, p, q] = cond_ex(W_hat.squeeze()** (-1 -gamma)  * Re[p, t, :] * Re[q, t, :],  basis) #, True)

        #Unconstrained optimum
        #x_opt = np.zeros((M, 2))
        #for m in range(M):
        #    x_opt[m] = 1/ (gamma * (W_grid[t - 1, j] + kappa))  * np.linalg.inv(b_hat[m]) @ a_hat[m]


        y_g = np.zeros((G, M))
        for g in range(G):
            y_g[g] = x_grid[g] @ a_hat.T - \
                0.5 * gamma * x_grid[g] @ b_hat @ x_grid[g].T * (W_grid[t - 1, j] + kappa)

        #x_opt =x_grid[np.argmax(y_g, axis = 0)].T
        x[j, :, t - 1] = x_grid[np.argmax(y_g, axis = 0)].T
        x[j, :, t:T] = x_hat                      #Store X_hat that was used!

stack_plot(x[5], T, "Income - 2nd order Taylor + discrete grid")


a =x[5]
