"""
Demonstrate equivalence of the one-shot LP and the iterative algorithms
outlined in [1] for minimal Robust Positively Invariant set generation.

Assumes that make_system generates a single system (no multiple Q's - but if
there are, the first one in the list is taken and the rest are ignored).
So the comparison is done as iterative vs. one-shot.

[1] Trodden, "A One-Step Approach to Computing a Polytopic Robust
Positively Invariant Set", 2016.

Author: Danylo Malyuta.
"""

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import make_system as sys
import invariance_tools

matplotlib.rcParams.update({'font.size': 13})
matplotlib.rc('text', usetex=True)

np.random.seed(101)

# Compute the invariant sets
X = []
iterative_options = [False,True]
runtimes = []
for i in range(len(iterative_options)):
    t = time.time()
    X.append(invariance_tools.minRPI(sys.A_cl[0],sys.D,sys.R,sys.r,
                                     iterative=iterative_options[i]))
    elapsed = time.time()-t
    runtimes.append(elapsed)

color = ['red','blue','green']
linestyles = ['-','--',':']
fig = plt.figure(1,figsize=(7,4.5))
plt.clf()
ax = fig.add_subplot(221)
for i in range(len(X)):
    X[i].plot(ax,coords=[0,1],facecolor='none',edgecolor=color[i],linewidth=2,linestyle=linestyles[i])
ax.set_xlabel('$x$ position [m]')
ax.set_ylabel('$y$ position [m]')
ax = fig.add_subplot(224)
for i in range(len(X)):
    X[i].plot(ax,coords=[2,3],facecolor='none',edgecolor=color[i],linewidth=2,linestyle=linestyles[i])
ax.set_xlabel('$v_x$ velocity [m/s]')
ax.set_ylabel('$v_y$ velocity [m/s]')
ax = fig.add_subplot(222)
for i in range(len(X)):
    X[i].plot(ax,coords=[0,2],facecolor='none',edgecolor=color[i],linewidth=2,linestyle=linestyles[i])
ax.set_xlabel('$x$ position [m]')
ax.set_ylabel('$v_x$ velocity [m/s]')
ax = fig.add_subplot(223)
for i in range(len(X)):
    X[i].plot(ax,coords=[1,3],facecolor='none',edgecolor=color[i],linewidth=2,linestyle=linestyles[i])
ax.set_xlabel('$y$ position [m]')
ax.set_ylabel('$v_y$ velocity [m/s]')
plt.tight_layout()
plt.show()

fig.savefig('figures/feanicses_oneshot_invariant_set_vs_iterative.pdf',
            bbox_inches='tight', format='pdf', transparent=True)