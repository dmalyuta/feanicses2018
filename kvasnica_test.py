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

#%% Startup the MATLAB engine

try:
    meng
except NameError:
    print "Starting the MATLAB engine"
    meng = matlab.engine.start_matlab()
else:
    meng.eval("clear",nargout=0)
  
#%% System definition
    
A = sys.A
B = sys.B

#%% Function definitions

def mset(vars_matlab,vars_python):
    """
    Set matrices in MATLAB workspace.
    
    Parameters
    ----------
    vars_matlab : str or list(str)
        A string or a list of strings for the variables to set in the MATLAB
        workspace.
    vars_python : list(array)
        Numpy arrays that the MATLAB variables should be assigned to equal.
    """
    if type(vars_matlab) is not list:
        vars_matlab = [vars_matlab]
        vars_python = [vars_python]
    for var_matlab,var_python in zip(vars_matlab,vars_python):
        if len(var_python.shape)==1:
            # Make column vectors out of 1D array
            var_python = np.mat(var_python).T
        meng.workspace[var_matlab] = matlab.double(var_python.tolist())
        
def mget(var_matlab):
    """
    Retrieve an array from MATLAB workspace.
    
    Parameters
    ----------
    var_matlab : str
        Variable name in MATLAB workspace that is to be retrieved.
        
    Returns
    -------
    var_python : array
        Numpy array set to the value of the desired MATLAB workspace variable.
    """
    var_python = np.array(meng.eval("%s" % (var_matlab)))
    return var_python

def meval(expr):
    """
    Evaluate expression given by expr in MATLAB.
    
    Parameters
    ----------
    expr : str
        MATLAB expression to evaluate (just as you would type in the MATLAB
        command prompt).
    """
    meng.eval(expr,nargout=0)

#%% Compute maxCRPI set
    
# Nullspace of C (output selection matrix) as {x : nullC_A*x <= nullC_b}
# polytope
nullC_A = np.vstack((sys.C,-sys.C))
nullC_b = np.zeros((sys.C.shape[0]*2))
pinvC = la.pinv(sys.C) # Pseudoinverse of C

# We do not care about mass invariance, thus we remove the mass part of the
# system dynamics be relying on the fact that in the A matrix, the mass
# dynamics are decoupled from the x,y dynamics
mset(['A','B','D','C'],[sys.A,sys.B,sys.D,sys.C])
mset(['G','g','H','h','R','r'],[sys.G,sys.g,sys.H,sys.h,sys.R,sys.r])
mset(['nullC_A','nullC_b','pinvC'],[nullC_A,nullC_b,pinvC])

meval("Y = Polyhedron(G,g)")
meval("U = Polyhedron(H,h)")
meval("P = Polyhedron(R,r)")
meval("nullC = Polyhedron(nullC_A,nullC_b)")
meval("Omega = Y")
iter_count = 0
while True:
    iter_count += 1
    print iter_count
    meval("pre = ((Y-((C*D)*P))+((-C*B)*U)+((-C*A)*nullC))*(C*A*pinvC)")
    meval("Omega_next = pre & Omega")
    meval("stop = Omega_next==Omega")
    stop = meng.eval("stop")
    if stop:
        break
    else:
        meval("Omega = Omega_next")
X_inf = poly.Polytope(mget("Omega.A"),mget("Omega.b"))

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