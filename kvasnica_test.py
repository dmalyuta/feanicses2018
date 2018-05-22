"""
Test code for maxCRPI set synthesis.

Author: Danylo Malyuta.
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import matlab.engine
import matplotlib.pyplot as plt

import make_system as sys
import polytope as poly
import invariance_tools

#%% Startup the MATLAB engine

try:
    meng
except NameError:
    print "Starting the MATLAB engine"
    meng = matlab.engine.start_matlab()
else:
    meng.eval("clear",nargout=0)
  
#%% Compute maxCRPI set
    
X_inf = invariance_tools.maxCRPI(sys.A,sys.B,sys.D,sys.C,sys.G,sys.g,sys.H,sys.h,sys.R,sys.r,
                meng=meng)
## Nullspace of C (output selection matrix) as {x : nullC_A*x <= nullC_b}
## polytope
#nullC_A = np.vstack((sys.C,-sys.C))
#nullC_b = np.zeros((sys.C.shape[0]*2))
#pinvC = la.pinv(sys.C) # Pseudoinverse of C
#
## We do not care about mass invariance, thus we remove the mass part of the
## system dynamics be relying on the fact that in the A matrix, the mass
## dynamics are decoupled from the x,y dynamics
#mset(['A','B','D','C'],[sys.A,sys.B,sys.D,sys.C])
#mset(['G','g','H','h','R','r'],[sys.G,sys.g,sys.H,sys.h,sys.R,sys.r])
#mset(['nullC_A','nullC_b','pinvC'],[nullC_A,nullC_b,pinvC])
#
#meval("Y = Polyhedron(G,g)")
#meval("U = Polyhedron(H,h)")
#meval("P = Polyhedron(R,r)")
#meval("nullC = Polyhedron(nullC_A,nullC_b)")
#meval("Omega = Y")
#iter_count = 0
#while True:
#    iter_count += 1
#    print iter_count
#    meval("pre = ((Y-((C*D)*P))+((-C*B)*U)+((-C*A)*nullC))*(C*A*pinvC)")
#    meval("Omega_next = pre & Omega")
#    meval("stop = Omega_next==Omega")
#    stop = meng.eval("stop")
#    if stop:
#        break
#    else:
#        meval("Omega = Omega_next")
#X_inf = poly.Polytope(mget("Omega.A"),mget("Omega.b"))

color = ['red','blue','green']
linestyles = ['-','--',':']
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

fig.savefig('figures/maxcrpi.pdf', bbox_inches='tight', format='pdf', transparent=True)