"""
Demonstrate maximal Robust Controlled Positively Invariant set computation
algorithm [1] (Section 4). In this case the algorith is slightly modified to
handle output-invariance rather than full state invariance (i.e. we can select
particular states to maintain invariant and ignore the rest).

[1] Kvasnica et al., "Reachability Analysis and Control Synthesis for
Uncertain Linear Systems in MPT", 2015.

Author: Danylo Malyuta.
"""

import matlab.engine
import matplotlib.pyplot as plt

import make_system as sys
import invariance_tools

# Startup the MATLAB engine
try:
    meng
except NameError:
    print "Starting the MATLAB engine"
    meng = matlab.engine.start_matlab()
else:
    meng.eval("clear",nargout=0)
  
# Compute maxCRPI set
X_inf = invariance_tools.maxCRPI(sys.A,sys.B,sys.D,sys.C,
                                 sys.G,sys.g,sys.H,sys.h,sys.R,sys.r,
                                 meng=meng)

fig = plt.figure(1,figsize=(7,4.5))
plt.clf()
ax = fig.add_subplot(221)
X_inf.plot(ax,coords=[0,1],facecolor='none',edgecolor='red',linewidth=2)
ax.set_xlabel('$x$ position [m]')
ax.set_ylabel('$y$ position [m]')
ax = fig.add_subplot(222)
X_inf.plot(ax,coords=[0,2],facecolor='none',edgecolor='red',linewidth=2)
ax.set_xlabel('$x$ position [m]')
ax.set_ylabel('$v_x$ velocity [m/s]')
ax = fig.add_subplot(223)
X_inf.plot(ax,coords=[1,3],facecolor='none',edgecolor='red',linewidth=2)
ax.set_xlabel('$y$ position [m]')
ax.set_ylabel('$v_y$ velocity [m/s]')
ax = fig.add_subplot(224)
X_inf.plot(ax,coords=[2,3],facecolor='none',edgecolor='red',linewidth=2)
ax.set_xlabel('$v_x$ velocity [m/s]')
ax.set_ylabel('$v_y$ velocity [m/s]')
plt.tight_layout()
plt.show()

fig.savefig('figures/maxcrpi.pdf',
            bbox_inches='tight', format='pdf', transparent=True)