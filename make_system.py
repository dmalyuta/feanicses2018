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
from scipy.linalg import solve_discrete_are as dare
import matplotlib
import matplotlib.pyplot as plt

import polytope as poly

matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 15})

#%% Parameters

g = -3.7114 # [m/s^2] Aceleration due to gravity
g_e = 9.81 # [m/s^2] Magnitude of Earth's gravity
m_wet = 1905. # [kg] Wet mass (i.e. current mass)
I_sp = 225 # [s] Rocket engine specific impulse
n_engine = 6 # Number of rocket engines
phi = 27.*np.pi/180. # [rad] Rocket engine tilt
alpha_max = 15.*np.pi/180. # [rad] Maximum vehicle tilt angle
T_max = 3.1e3 # [N] Maximum thrust from a single rocket engine
dt = 1/25. # [s] Discretization time step
tau = 0.3 # Fraction of T_max for rocket engine throttling

alpha = 1./(I_sp*g_e*np.cos(phi)) # Coefficient of mass depletion dynamics

#%% Create linearized continuous-time system

x_eq = {'x': 0., 'y': 0., 'v_x': 0., 'v_y': 0., 'm': m_wet}
u_eq = {'F_x': 0., 'F_y': 0.}
p_eq = {'Q_x': 0., 'Q_y': 0.}

_denom = np.sqrt(u_eq['F_x']**2+(u_eq['F_y']-x_eq['m']*g)**2)
A_c = np.array([[0.,0.,1.,0.,0.],
                [0.,0.,0.,1.,0.],
                [0.,0.,0.,0.,-(u_eq['F_x']+p_eq['Q_x'])/x_eq['m']**2],
                [0.,0.,0.,0.,-(u_eq['F_y']+p_eq['Q_y'])/x_eq['m']**2],
                [0.,0.,0.,0.,alpha*g*(u_eq['F_y']-x_eq['m']*g)/_denom]])

B_c = np.array([[0.,0.],
                [0.,0.],
                [1./x_eq['m'],0.],
                [0.,1./x_eq['m']],
                [-alpha*u_eq['F_x']/_denom,
                 -alpha*(u_eq['F_y']-x_eq['m']*g)/_denom]])

D_c = np.array([[0.,0.],
                [0.,0.],
                [1./x_eq['m'],0.],
                [0.,1./x_eq['m']],
                [0.,0.]])

# Dimensions
n = 5 # State
m = 2 # Input
d = 2 # Disturbance

#%% Discretize the system

# Method: https://en.wikipedia.org/wiki/Discretization#cite_ref-1

M = sla.expm(np.block([[A_c,B_c],[np.zeros((m,n)),np.zeros((m,m))]])*dt)
A = M[0:n,0:n]
B = M[0:n,n:]

M = sla.expm(np.block([[A_c,D_c],[np.zeros((d,n)),np.zeros((d,d))]])*dt)
D = M[0:n,n:]

C = np.hstack((np.eye(4),np.zeros((4,1))))

#%% Make a stabilizing LQR controller

Dx = np.diag([1,1,0.05,0.05,0.1])
Du = np.diag([n*T_max*np.cos(phi)*np.sin(alpha_max),
              np.maximum(np.abs(n*T_max*np.cos(phi)*np.cos(alpha_max)+m_wet*g),
                         np.abs(n*tau*T_max*np.cos(phi)+m_wet*g))])

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
    plt.show()
    fig.savefig('figures/root_locus.pdf', bbox_inches='tight', format='pdf', transparent=True)
    
#%% Specification polytopes

# Construct admissible input polytope
# Lower right corner
x_1 = n*tau*T_max*np.cos(phi)*np.sin(alpha_max)
y_1 = n*tau*T_max*np.cos(phi)+m_wet*g
# Upper right corner
x_2 = n*T_max*np.cos(phi)*np.sin(alpha_max)
y_2 = n*T_max*np.cos(phi)*np.cos(alpha_max)+m_wet*g
# Construct H-rep representation of the polytope
# Facet normals
a_top = np.array([0,1])
a_bottom = np.array([0,-1])
a_right = np.array([y_2-y_1,x_1-x_2])
a_left = np.array([y_1-y_2,x_1-x_2])
# Facet distances
b_top = np.abs(y_2)
b_bottom = np.abs(y_1)
b_right = np.abs(np.inner(a_right,np.array([x_1,y_1])))
b_left = np.abs(np.inner(a_left,np.array([-x_1,y_1])))
# Construct polytope
H = np.vstack((a_top,a_right,a_bottom,a_left))
h = np.array([b_top,b_right,b_bottom,b_left])
# Make the polytope
U = poly.Polytope(H,np.mat(h).T)

if False:
    fig= plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111)
    U.plot(ax,facecolor='none',edgecolor='red',linewidth=2)
    ax.axis('equal')
    ax.set_xlabel('$T_x$ [N]')
    ax.set_ylabel('$T_y$ [N]')
    ax.set_title('$\mathcal U$')
    plt.show()
    
    fig.savefig('figures/feanicses_U_set.pdf',
                bbox_inches='tight', format='pdf', transparent=True)

# Construct safe states (outputs) polytope
pos_err_max = (0.5,0.1) # [m] Max tolerated (horizontal,vertical) position error
vel_err_max = (0.5,0.01) # [m/s] Max tolerated (horizontal,vertical) velocity error
Y = poly.Polytope(R=[(-pos_err_max[0],pos_err_max[0]),
                     (-pos_err_max[1],pos_err_max[1]),
                     (-vel_err_max[0],vel_err_max[0]),
                     (-vel_err_max[1],vel_err_max[1])])
G,g = Y.P, Y.p

# Construct polytope of possible disturbances
disturbance_max = (400,400) # [N] (horizontal,vertical) disturbance force
P = poly.Polytope(R=[(-disturbance_max[0],disturbance_max[0]),
                     (-disturbance_max[1],disturbance_max[1])])
R,r = P.P, P.p
