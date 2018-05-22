"""
Demonstrate one-shot LP algorithm [1] for minimal Robust Positively Invariant
set generation.

Assumes that make_system.py generates a system with one R and up to 3
different Q weights for the LQR controller.
So the comparison is done as invariant set sythesized by one-shot LP vs.
Q weight in the LQR problem.

[1] Trodden, "A One-Step Approach to Computing a Polytopic Robust
Positively Invariant Set", 2016.

Author: Danylo Malyuta.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import make_system as sys
import invariance_tools

matplotlib.rcParams.update({'font.size': 13})
matplotlib.rc('text', usetex=True)

np.random.seed(101)

# Generate a template for the invariant polytope
n_extra = 0 # Number of extra random facet normals
# Generate random facet normals via
# https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
angles = np.hstack((np.random.uniform(low=0., high=np.pi, size=(n_extra,3)),
                    np.random.uniform(low=0., high=2*np.pi, size=(n_extra,1))))
G_extra = np.zeros((n_extra,5))
for i in range(angles.shape[0]):
    phi = angles[i]
    G_extra[i] = np.array([np.cos(phi[0]),
           np.sin(phi[0])*np.cos(phi[1]),
           np.sin(phi[0])*np.sin(phi[1])*np.cos(phi[2]),
           np.sin(phi[0])*np.sin(phi[1])*np.sin(phi[2])*np.cos(phi[3]),
           np.sin(phi[0])*np.sin(phi[1])*np.sin(phi[2])*np.sin(phi[3])])
G = []
for A_cl in sys.A_cl:
    Gi = invariance_tools.generateTemplate(A_cl,sys.D,sys.R,sys.r)
    Gi = np.vstack((Gi,G_extra)) # Add the extra random facets
    G.append(Gi)

# Compute the invariant sets
X = []
for i in range(len(sys.A_cl)):
    X.append(invariance_tools.trodden(sys.A_cl[i],sys.D,sys.R,sys.r,G=G[i]))

color = ['red','blue','green']
fig = plt.figure(1,figsize=(7,4.5))
plt.clf()
ax = fig.add_subplot(221)
for i in range(len(X)):
    X[i].plot(ax,coords=[0,1],facecolor='none',edgecolor=color[i],linewidth=2)
ax.set_xlabel('$x$ position [m]')
ax.set_ylabel('$y$ position [m]')
ax = fig.add_subplot(224)
for i in range(len(X)):
    X[i].plot(ax,coords=[2,3],facecolor='none',edgecolor=color[i],linewidth=2)
ax.set_xlabel('$v_x$ velocity [m/s]')
ax.set_ylabel('$v_y$ velocity [m/s]')
ax = fig.add_subplot(222)
for i in range(len(X)):
    X[i].plot(ax,coords=[0,2],facecolor='none',edgecolor=color[i],linewidth=2)
ax.set_xlabel('$x$ position [m]')
ax.set_ylabel('$v_x$ velocity [m/s]')
ax = fig.add_subplot(223)
for i in range(len(X)):
    X[i].plot(ax,coords=[1,3],facecolor='none',edgecolor=color[i],linewidth=2)
ax.set_xlabel('$y$ position [m]')
ax.set_ylabel('$v_y$ velocity [m/s]')
plt.tight_layout()
plt.show()

fig.savefig('figures/oneshot_invariant_set.pdf', bbox_inches='tight', format='pdf', transparent=True)