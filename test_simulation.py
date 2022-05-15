# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:52:15 2022

@author: Swen
"""

from KNW import *

x0 = np.array([0.0181, -0.0063, 0.0014, 0.0240, -0.0148, 0.0053, 0.08, -0.19, 0.35,
       0.0002, -0.0001, 0.0061,  0.0452, -0.0053, -0.0076, -0.0211, 0.1659,
       0.403, 0.039, 0.149, -0.381, 0.089, -0.083, 0.01, 0.01, 0.01, 0.01])

pars_0 = KNW_pars(x0)
T = 20
#T = 1000
M = 10000

pars_0
#With original parameters
knw_0 = KNW(pars_0)
res = knw_0.simulate(T, M, 'EM')

np.diff(res.log_S)


plt.plot(res.S)

a = np.log(res.S)
a = np.diff(res.log_S, axis = 0)

a.sort()
a = a.T

x = x0
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

#Other Lambda0/Lambda 1 follow from restriction
Lambda0 = np.concatenate([Lambda0, np.zeros(2)])
Lambda1 = np.concatenate([Lambda1, np.zeros((2,2))])

#Compute the 4th row of Lambda
Lambda0[3] = (eta_s - np.dot(sigma_s[0:2], Lambda0[0:2])) \
                / sigma_s[3]
Lambda1[3] = - np.dot(sigma_s[0:2], Lambda1[0:2]) /sigma_s[3]

Sigma_x = np.concatenate((np.identity(2), np.zeros((2,2))))

M  = 10000
T = 20
delta = 0.1
nsteps = int(T/delta)

Z = np.random.normal(0, 1, size = (4, nsteps, M))


X = np.zeros((2, nsteps + 1, M))

for t in range(nsteps):
    #t = 0
    dX = -K @ X[:, t, :] * delta  + Sigma_x.T @ Z[:, t, :] * np.sqrt(delta)
    X[:, t + 1, :] = X[:, t, :] + dX

plt.plot(X[0, :, 1:100])
plt.plot(X[1, :, 1:100])

plt.plot(res.X1[:, 1:100])
plt.plot(res.X2[:, 1:100])



r = R0 + np.inner(R1.T, X.T)

pi = delta_0pi + np.inner(delta_1pi, X.T)

plt.plot(r.T)
plt.plot(pi.T)


#Inflation
dpi = pi[:, 1:] * delta + np.inner(sigma_pi , Z.T) * np.sqrt(delta)
Pi = np.cumprod(1 + dpi, axis = 1)

plt.plot(Pi.T)
res.plot('CPI')

np.mean(Pi[:, 199])
np.mean(res.CPI[19])

vol(Pi[:, 199])
vol(res.CPI[:, 19])

#Stock returns
dS = (r[:, 1:] + eta_s) * delta + np.inner(sigma_s, Z.T) *np.sqrt(delta)
S  = np.cumprod(1 + dS, axis = 1)
plt.plot(S.T)
res.plot('S')

np.mean(S[:, 199])
np.mean(res.S[19])
vol(S[:, 199])
vol(res.S[:, 19])



a =  np.inner(R1.T, X.T)
vol(X[0])
vol(res.X1)
vol(X[1])
vol(res.X2)
