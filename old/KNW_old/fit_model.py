# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 11:24:17 2022

@author: Swen
"""
import numpy as np
from numba import njit

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



x0 = np.array([0.0181, -0.0063, 0.0014, 0.0240, -0.0148, 0.0053, 0.08, -0.19, 0.35,
      0.0002, -0.0001, 0.0061,  0.0452, -0.0053, -0.0076, -0.0211, 0.1659,
      0.403, 0.039, 0.149, -0.381, 0.089, -0.083, 1, 1, 1])

x = x0





def fit_model(data):
    """
        Fits the model to the input data

        Returns:
            fitted model
    """




@njit
def BN(M, tau, R1):
    #IMPORTANT: CHECK THIS FORMULA, MIGHT BE WRONG
    n = len(tau)
    B = np.zeros((2, n))
    Minv = np.linalg.inv(M)
    for t in range(n):
        B[:, t] = Minv @ (expm(-M  * tau[t]) - np.identity(2)) @R1
    return(B)

#M = (K.T + Lambda1.T)
tau = np.array([1, 2, 3, 5, 10])
B = BN(M, tau, R1)

B/tau

@timer
def func():
    for _ in range(10000):
        BN(M, tau, R1)
func()


#@njit
def AN(M, tau, delta, Lambda0, R0, R1):
    """
        M = pricing kernel
        tau = tau's to compute
        delta = integration stepsize

        Note: This can probably be done analytically!!
    """
    #delta = 0.1
    T = tau[-1]
    Tau = np.arange(0, T, delta)

    B = BN(M, Tau, R1)

    y = -Lambda0.T @ B + np.sum(B * B, axis = 0)

    idx = tau // delta
    idx = idx.astype(int)
    n = len(tau)
    A = np.zeros(n)
    for t in range(n):
       A[t] = -R0 * tau[t] + np.trapz(y[:idx[t]], dx=delta) #Integrate y

    return A


A = AN(M, tau, 0.1, Lambda0, R0, R1)

@timer
def func():
    for _ in range(10000):
        AN(M, tau, 0.1, Lambda0, R0, R1)
func()


def maximum_likelihood(x):
    """
        Maximum likelihood function

        Input:
            x   Vector of parameters 23 in total
    """
    #x = np.array([i for i in range(28)])

    #Extract parameters from x vector!
    delta_0pi = x[0]
    delta_1pi = x[[1, 2]]

    R0 = x[3]
    R1 = x[[4, 5]]


    K = np.zeros([2, 2])
    K[0, 0] = x[6]
    K[1, 0] = x[7]
    K[1, 1] = x[8]

    sigma_pi = np.concatenate([x[9:12], np.zeros(1)])
    eta_s = x[12]
    sigma_s = x[13:17]

    Lambda0 = x[17:19]
    Lambda1 = x[19:23].reshape(2,2)

    sigma_v = np.diag(x[23:26])
    sigma_v = sigma_v.astype(float)

    yields = np.array(data)[:, 0:5]

    #Compute A and B

    tau = np.array([1, 2, 3, 5, 10])
    M = K.T + Lambda1.T


    B = BN(M, tau, R1)
    A = AN(M, tau, 0.1, Lambda0, R0, R1)

    #B/ tau
    #A/ tau

    #X = Y[0:2, :, 0].squeeze() #Example: 1 trajectory
    # b=  - X.T @ B
    #yn = (-A -  X.T @ B)/ tau

    T = len(yields)
    idx = np.array([1, 3]) #2Y and 5Y are observed without error

    #Compute X's
    y = yields[:, idx]
    a = A[idx]
    b = B[:, idx].T  #transpose

    l3 = -0.5 * T * np.log(np.linalg.det(b))

    X = -( y * tau[idx] + a) @ np.linalg.inv(b)

    #plt.plot(yields)
    #plt.plot(X) #NOTE: x's are completely off...l B calculation is wrong!


    l1= L1(X, A, B, yields, sigma_v, tau)

    """ Determine L2"""

    Y = np.concatenate([X, data[[ 'Inflation','^RUI']]], axis = 1)
    theta0 = np.array([0,
                       0,
                       delta_0pi - 0.5 * sigma_pi @ sigma_pi.T,
                       R0 + eta_s - 0.5 * sigma_s @ sigma_s.T])
    theta1 = np.array([-K[0],
                       -K[1],
                       delta_1pi.T,
                       R1.T])

    theta1 = np.concatenate((theta1, np.zeros((4,2))), axis =1)

    Sigma_y= np.concatenate((np.identity(2), np.zeros((2,2))), axis = 1)
    Sigma_y = np.concatenate([Sigma_y, sigma_pi.reshape(1, 4), sigma_s.reshape(1, 4)])

    d, U = np.linalg.eig(theta1)

    U_inv = np.linalg.inv(U)

    h = 1/12
    gamma_h = U @ np.diag(np.exp(d * h )) @ U_inv
    F = h * np.diag( alpha(d * h))
    mu_h = U @ F @ U_inv @ theta0

    inner = np.add.outer(d, d) * h
    V = U_inv @ Sigma_y @ Sigma_y.T @ U_inv.T * h *[alpha(v) for v in inner]
    sigma_h = U @ V @ U.T


    """  IMPORTANT: Y is length 220. This one misses 1!"""
    #b = Y[:-1] @ gamma_h
    epsilon = Y[1:] - mu_h  -  Y[:-1] @ gamma_h

    l2 = -0.5 * T * np.linalg.det(sigma_h) \
        + 0.5 * np.sum([eps @ np.linalg.inv(sigma_h) @ eps.T for eps in epsilon])

    loglik = l1 + l2 + l3
    return loglik

maximum_likelihood(x = x0)


@timer
def func():
    for _ in range(10000):
        maximum_likelihood(x = x0)
func()


@njit
def alpha(X):
    return [(np.exp(x) - 1) / x if x != 0 else 1 for x in X]

#Run once to compile!
alpha(d * h)

@timer
def func():
    for _ in range(10000):
        alpha(d * h)

func()


@njit
def L1(X, A, B, yields, sigma_v, tau):
    """
        A:      6x1 params for different maturities
        B:      6x2 params for different maturities
        yields: 6xT vector of yields
    """

    #import matplotlib.pyplot as plt
    #plt.plot(X)

    #Other yields are observed with a meassurement error:
    idx = np.array([0, 2, 4])
    y = yields[:, idx]
    a = A[idx]
    b = B[:, idx]
    v =   y + ( a + X @ b) / tau[idx]

    loglik = -T / 2 * np.log(np.linalg.det(sigma_v))  \
       - 1 / 2 * np.sum(np.array([w @ np.linalg.inv(sigma_v) @ w.T for w in v]))
    return loglik

#Compile once!
L1(X, A, B, yields, sigma_v, tau)

@timer
def func():
    for _ in range(10000):
        L1(X, A, B, yields, sigma_v, tau)

func()


def L2(Y, mu, gamma, Sigma):
    Y =



    eps = Y[1:] - mu - gamma * Y[]
