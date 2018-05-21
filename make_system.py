"""
Create the skycrane system.

Numerical data taken from [1].

[1] Behcet Acikmese and Scott R. Ploen.  "Convex Programming Approach to
Powered Descent Guidance for Mars Landing", Journal of Guidance, Control, and
Dynamics, Vol. 30, No. 5 (2007), pp. 1353-1366.

Author: Danylo Malyuta.
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.io as sio
from scipy.linalg import solve_discrete_are as dare
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 20})

#%% Parameters

g = -3.7114 # [m/s^2] Aceleration due to gravity
g_e = 9.81 # [m/s^2] Magnitude of Earth's gravity
I_sp = 225 # [s] Rocket engine specific impulse
n_engine = 6 # Number of rocket engines
phi = 27.*np.pi/180. # [rad] Rocket engine tilt
alpha_max = 15.*np.pi/180. # [rad] Maximum vehicle tilt angle
T_max = 3.1e3 # [N] Maximum thrust from a single rocket engine
dt = 1/25. # [s] Discretization time step

alpha = 1./(I_sp*g_e*np.cos(phi)) # Coefficient of mass depletion dynamics

#%% Create linearized continuous-time system

x_eq = {'x': 0., 'y': 0., 'v_x': 0., 'v_y': 0., 'm': 1905.}
u_eq = {'F_x': 0., 'F_y': 0.}

_denom = np.sqrt(u_eq['F_x']**2+(u_eq['F_y']-x_eq['m']*g)**2)
A_c = np.array([[0.,0.,1.,0.,0.],
                [0.,0.,0.,1.,0.],
                [0.,0.,0.,0.,-u_eq['F_x']/x_eq['m']**2],
                [0.,0.,0.,0.,-u_eq['F_y']/x_eq['m']**2],
                [0.,0.,0.,0.,alpha*g*(u_eq['F_y']-x_eq['m']*g)/_denom]])

B_c = np.array([[0.,0.],
                [0.,0.],
                [1./x_eq['m'],0.],
                [0.,1./x_eq['m']],
                [-alpha*u_eq['F_x']/_denom,
                 -alpha*(u_eq['F_y']-x_eq['m']*g)/_denom]])

# Dimensions
n = 5 # State
m = 2 # Input

#%% Discretize the system

# Method: https://en.wikipedia.org/wiki/Discretization#cite_ref-1

M = sla.expm(np.block([[A_c,B_c],[np.zeros((m,n)),np.zeros((m,m))]])*dt)

A = M[0:n,0:n]
B = M[0:n,n:]

#%% Make a stabilizing LQR controller

Dx = np.diag([1,1,0.05,0.05,0.1])
Du = np.diag([n*T_max*np.cos(phi)*np.sin(alpha_max),n*T_max*np.cos(phi)])

# Unscaled weight
Iq = np.eye(n)
Iq[-1,-1] = 0.
Qhat = [Iq,10*Iq]
Rhat = np.eye(m)

# Scaled weights
Q = [la.inv(Dx.T).dot(Qhat_i).dot(la.inv(Dx)) for Qhat_i in Qhat]
R = la.inv(Du.T).dot(Rhat).dot(la.inv(Du))

K, P, A_cl = [], [], []
for Qi in Q:
    P.append(dare(A,B,Qi,R))
    K.append(-la.inv(R+B.T.dot(P[-1]).dot(B)).dot(B.T).dot(P[-1]).dot(A))
    A_cl.append(A+B.dot(K[-1]))

if False:
    # Plot closed-loop eigenvalues
    spec_cl = [la.eigvals(A_cl_i) for A_cl_i in A_cl]
    fig = plt.figure(1,figsize=(6,6))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.add_artist(plt.Circle((0, 0), 1, facecolor='none', edgecolor='black'))
    for i in range(len(spec_cl)):
        ax.plot(np.real(spec_cl[i]), np.imag(spec_cl[i]),
                marker='x', linestyle='none', label=('$Q=%d\cdot I$'%(int(np.max(Qhat[i])))))
    ax.axis('equal')
    ax.set_xlim([-1.2,1.2])
    ax.set_ylim([-1.2,1.2])
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.legend()
    fig.show()
    fig.savefig('figures/root_locus.pdf', bbox_inches='tight', format='pdf', transparent=True)