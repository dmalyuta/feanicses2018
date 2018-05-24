"""
Test of disturbance-free invariant set computation according to [1].

[1] Blanco and De Moor, "Robust MPC based on symmetric low-complexity
polytopes", 2009.

Author: Danylo Malyuta.
"""

import numpy as np
import numpy.linalg as la
import cvxpy as cvx

import matplotlib.pyplot as plt

import make_system as sys
import polytope as poly

# Make system
A = sys.A
B = sys.B
K = sys.K[0]

phi = (A+B.dot(K))[:4,:4]

spec_phi, W = la.eig(phi)

# Compute the invariant set in the disturbance-free case
V_f = la.inv(W)
n_v = V_f.shape[0]

X_f = poly.Polytope(np.vstack((V_f,-V_f)),np.ones((2*n_v,1)))
P_f, p_f = np.array(X_f.P), np.array(X_f.p).flatten()

def find_rho(P,p,solver):
    rho = cvx.Variable()
    Y = cvx.Variable(p.size,p_f.size)
    cost = cvx.Minimize(rho)
    constraints = [Y*p_f <= rho*p,
                   Y*P_f == P,
                   Y >= 0]
    problem = cvx.Problem(cost, constraints)
    optimal_value = problem.solve(solver=solver, verbose=True)
    return rho.value

rho_x = find_rho(sys.Y.P, sys.Y.p, solver=cvx.ECOS)
rho_y = find_rho(sys.U.P.dot(K[:,:4]), sys.U.p, solver=cvx.GUROBI)
rho = np.maximum(rho_x,rho_y)
X_f = poly.Polytope(np.vstack((rho*V_f,-rho*V_f)),np.ones((2*n_v,1)))

# Plot

fig = plt.figure(1,figsize=(7,4.5))
plt.clf()
ax = fig.add_subplot(221)
X_f.plot(ax,coords=[0,1],facecolor='none',edgecolor='red',linewidth=2)
ax.set_xlabel('$x$ position [m]')
ax.set_ylabel('$y$ position [m]')
ax = fig.add_subplot(222)
X_f.plot(ax,coords=[0,2],facecolor='none',edgecolor='red',linewidth=2)
ax.set_xlabel('$x$ position [m]')
ax.set_ylabel('$v_x$ velocity [m/s]')
ax = fig.add_subplot(223)
X_f.plot(ax,coords=[1,3],facecolor='none',edgecolor='red',linewidth=2)
ax.set_xlabel('$y$ position [m]')
ax.set_ylabel('$v_y$ velocity [m/s]')
ax = fig.add_subplot(224)
X_f.plot(ax,coords=[2,3],facecolor='none',edgecolor='red',linewidth=2)
ax.set_xlabel('$v_x$ velocity [m/s]')
ax.set_ylabel('$v_y$ velocity [m/s]')
plt.tight_layout()
plt.show()
