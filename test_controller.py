"""
Test controller(s).

Author: Danylo Malyuta.
"""

import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import cvxpy as cvx

import make_system as sys
import controllers as ctrl

matplotlib.rcParams.update({'font.size': 13})
matplotlib.rc('text', usetex=True)

np.random.seed(101)

#%% Create controllers

controller_type = 'lfs'

if controller_type == 'lqr':
    # LQR controller
    n,m = sys.B.shape
    # Unscaled weights
    Iq = np.eye(n)
    Iq[-1,-1] = 0.
    Qhat = Iq
    Rhat = np.eye(m)
    # Scaled weights
    Q = la.inv(sys.Dx.T).dot(Qhat).dot(la.inv(sys.Dx))
    R = la.inv(sys.Du.T).dot(Rhat).dot(la.inv(sys.Du))
    mu = ctrl.LQR(sys.A,sys.B,Q,R)
elif controller_type == 'lfs':
    # Linear From Specifications controller
    mu = ctrl.LFS(sys.A,sys.B,sys.C,sys.D,sys.G,sys.g,sys.H,sys.h,sys.R,sys.r)
else:
    raise AssertionError("Unknown controller type (%s)" % (controller_type))

#%% Simulate the closed-loop system

class SimulationOutput:
    def __init__(self):
        self.t = [] # Times
        self.u = [] # States
        self.x = [] # Inputs
        self.p = [] # Noise
        
    def add(self,t,x,u,p):
        self.t.append(t)
        self.x.append(x)
        self.u.append(u)
        self.p.append(p)
        
    def concatenate(self):
        # NB: after calling concatenate, can no longer call add
        self.t = np.asarray(self.t)
        self.x = np.vstack(self.x)
        self.u = np.vstack(self.u)
        self.p = np.vstack(self.p)
        
sim_history = SimulationOutput()
T = 10. # [s] Simulation run-time
t, x = 0., np.zeros((n))
while t <= T:
    x_prev = np.copy(x)
    p = np.random.uniform(low=-np.array(sys.disturbance_max),
                          high=np.array(sys.disturbance_max))
    u = mu(x)
    x = sys.A.dot(x)+sys.B.dot(u)+sys.D.dot(p)
    sim_history.add(t,x_prev,u,p)
    t += sys.dt
sim_history.concatenate()

#%% Visualize

# States
fig = plt.figure(1,figsize=(7,4.5))
plt.clf()
ax = fig.add_subplot(221)
sys.Y.plot(ax,coords=[0,1],facecolor='none',edgecolor='red',linewidth=2)
ax.plot(sim_history.x[:,0],sim_history.x[:,1],color='black')
ax.set_xlabel('$x$ position [m]')
ax.set_ylabel('$y$ position [m]')
ax = fig.add_subplot(222)
sys.Y.plot(ax,coords=[0,2],facecolor='none',edgecolor='red',linewidth=2)
ax.plot(sim_history.x[:,0],sim_history.x[:,2],color='black')
ax.set_xlabel('$x$ position [m]')
ax.set_ylabel('$v_x$ velocity [m/s]')
ax = fig.add_subplot(223)
sys.Y.plot(ax,coords=[1,3],facecolor='none',edgecolor='red',linewidth=2)
ax.plot(sim_history.x[:,1],sim_history.x[:,2],color='black')
ax.set_xlabel('$y$ position [m]')
ax.set_ylabel('$v_y$ velocity [m/s]')
ax = fig.add_subplot(224)
sys.Y.plot(ax,coords=[2,3],facecolor='none',edgecolor='red',linewidth=2)
ax.plot(sim_history.x[:,2],sim_history.x[:,3],color='black')
ax.set_xlabel('$v_x$ velocity [m/s]')
ax.set_ylabel('$v_y$ velocity [m/s]')
plt.tight_layout()
plt.show()

fig.savefig(('figures/feanicses_controller_states_%s.pdf'%(controller_type)),
            bbox_inches='tight', format='pdf', transparent=True)

# Inputs
fig= plt.figure(2)
plt.clf()
ax = fig.add_subplot(111)
sys.U.plot(ax,facecolor='none',edgecolor='red',linewidth=2)
ax.plot(sim_history.u[:,0],sim_history.u[:,1],marker='x',linestyle='none')
ax.axis('equal')
ax.set_xlabel('$T_x$ [N]')
ax.set_ylabel('$T_y$ [N]')
ax.set_title('$\mathcal U$')
plt.show()

fig.savefig(('figures/feanicses_controller_inputs_%s.pdf'%(controller_type)),
            bbox_inches='tight', format='pdf', transparent=True)

# Disturbances
fig= plt.figure(3)
plt.clf()
ax = fig.add_subplot(111)
sys.P.plot(ax,facecolor='none',edgecolor='red',linewidth=2)
ax.plot(sim_history.p[:,0],sim_history.p[:,1],marker='x',linestyle='none')
ax.axis('equal')
ax.set_xlabel('$Q_x$ [N]')
ax.set_ylabel('$Q_y$ [N]')
ax.set_title('$\mathcal P$')
plt.show()
