"""
Functions for generating invariant sets.

Author: Danylo Malyuta.
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import cvxpy as cvx
import matlab.engine
import matplotlib.pyplot as plt

import polytope

def EigInWhichPoly(eigval,Nmax=1000,epsmin=0.):
    """
    **Translated from original MATLAB code by Dylan Janak**
    Finds a regular polygon in the complex plane that contains lambda = a+bi
    these polygons are inscribed in the unit circle and have a vertex at 1+0i
    N is the number of sides of the enclosing polygon
    eps>0: ball centered at lambda with radius "eps" is also in this polygon
    eps<0: lambda is in the polygon expanded by factor of [1+eps/cos(pi/N)]
    Nmax is the maximum number of sides the polygon is allowed to have
    
    Parameters
    ----------
    eigval : array or float
        1D array (or a single value) of eigenvalues.
    Nmax : int
        Maximum number of facets.
    epsmin : float
        Largest tolerated distance between eigenvalue and polytope edge.
        
    Returns
    -------
    N : array
        Set of edgec counts required for the respective eigenvalue in eigval.
    epsilon : float
        Distance between containing polytope and the corresponding eigenvalue
        in eigval.
    """
    if type(eigval) is not np.ndarray:
        eigval = np.array([eigval])
    
    m = eigval.size
    N = np.zeros((m),dtype=int)
    epsilon = np.zeros((m))
    
    for i in range(m):
        r = np.absolute(eigval[i])
        if r >= 1-epsmin:
            raise AssertionError("abs(lambda) must be < 1-epsmin")
        angle = lambda h: np.arctan2(np.imag(h),np.real(h))
        theta = angle(eigval[i])
        
        Ni = 2 # polygon must have at least 3 sides for interior to be non-empty
        eps = -np.inf # eps is how far lambda is from the outside the polygon
        while eps <= epsmin and Ni < Nmax:
            Ni += 1
            phi = np.pi/Ni
            D = theta%(2*phi)-phi
            eps = np.cos(phi)-r*np.cos(D)
        if Ni <= Nmax:
            N[i] = Ni
            epsilon[i] = eps
        else:
            N[i] = np.inf
            epsilon[i] = 1-r
    return N, epsilon

def realjordan(A):
    """
    Finds the real Jordan form of matrix A, so that AV = VC for some real
    Jordan matrix C.
    """
    c,V = la.eig(A)
    C = np.diag(c)
    # ensure eigenvalues are sorted properly
#    idx = np.real(c).argsort()[::-1]
#    C = np.diag(c[idx])
#    V = V[:,idx]
    
    if la.cond(V) > 1e6:
        # The nondiagonalizable case is not implemented due to numerical issues
        # with Jordan decomposition if elements of "A" are not ratios of small
        # integers.
        raise AssertionError("The matrix is too close to nondiagonalizable")
    else:
        n = C.shape[0]-1
        k = 0
        while k <= n:
            if np.imag(C[k,k]) == 0:
                k += 1
            else:
                V[:,k:k+2] = np.column_stack((np.real(V[:,k]),np.imag(V[:,k])))
                C[k:k+2,k:k+2] = np.array([[np.real(C[k,k]),np.imag(C[k,k])],
                                           [-np.imag(C[k,k]),np.real(C[k,k])]])
                k += 2
    return np.real(V), np.real(C)

def generateTemplate(A,E,F,f):
    """
    Generates an invariant polytope Gx<=g template matrix G for a stable system
    x+=Ax+Ew, Fw<=f.
    A must be DIAGONALIZABLE and have all eigenvalues in the open unit disk
    """
    n = A.shape[0]-1
    
    V,D = realjordan(A) # finds the real Jordan form of A
    
    k=0
    while k<n:
        isfirst = k==0
        if D[k,k+1] == 0:
            Gk = np.array([[1],[-1]])
            k += 1
        else:
            lambdak = D[k,k]+D[k,k+1]*1j
            Nk = EigInWhichPoly(lambdak,Nmax=1000,epsmin=1e-4)[0][0]
            phik = 2*np.pi/Nk
            Gk = np.zeros((Nk,2))
            for l in range(Nk):
                Gk[l,:] = np.array([np.cos(l*phik),np.sin(l*phik)])
            k += 2
        if isfirst:
            Gz = Gk
        else:
            Gz = sla.block_diag(Gz,Gk)
    if k==n:
        Gk = np.array([[1],[-1]])
        Gz = sla.block_diag(Gz,Gk)
    G = Gz.dot(la.inv(V))
    for i in range(G.shape[0]):
        G[i,:] = G[i,:]/la.norm(G[i,:],2)
    return G

def minRPI(A,D,R,r,G=None,iterative=False,cv_tol=1e-8,max_iter=np.inf,
           verbose=True):
    """
    Compute Minimal Robust Positively Invariant (RPI) set for the system
    
        x+ = A*x+D*p  where  p in {p : R*p<=r}
        
    Algorithm comes from [1].
    
    NB: the formulation is can be numerically badly conditions (e.g. the 
    current system in make_system.py fails to be solved well with ECOS, but
    GUROBI does the job). TODO: potentially introduce scaling?
    
    [1] Trodden, "A One-Step Approach to Computing a Polytopic Robust
    Positively Invariant Set", 2016.
    
    Inputs:
       A : (n,n) array
           Dynamics matrix. Must be STABLE and DIAGONALIZABLE.
       D : (n,d) array
           Matrix through which process noise acts.
       R : (n_r,d) array
           Facet normals of the process noise set.
       r : (n_r) array
           Facet distances of the process noise set.
       G : (n_g,n) array, optional
           Invariant polytope template (i.e. facet normals).
       iterative : bool, optional
           If ``True``, use the iterative rather than one-shot LP method for
           computing the invariant set's facet distances. In this case the
           cv_tol and max_iter arguments get used.
       cv_tol : float, optional
           Convergence tolerance for the iterative approach (when
           ``iterative==True``).
       max_iter : int, optional
           Maximum number of iterations for the iterative approach (when
           ``iterative==True``).
       verbose: bool, optional
           If ``True``, optimizer prints its output.
                       
    Outputs:
       X : Polytope
           The minimal RPI set.
    """
    def raiseError(error_string):
        raise AssertionError("[minRPI] %s" % (error_string))
    
    n = A.shape[0]
    n_r = D.shape[1]
    
    if G is None:
        G = generateTemplate(A,D,R,r) 
    n_g = G.shape[0]
    
    if not iterative:
        # Apply Trodden's LP [1]
        c = cvx.Variable(n_g)
        d = cvx.Variable(n_g)
        xi = cvx.Variable(n,n_g)
        omega = cvx.Variable(n_r,n_g)
        cost = cvx.Maximize(sum(c+d))
        constraints = []
        for i in range(n_g):
            constraints += [c[i] <= G[i].dot(A)*xi[:,i],
                            G*xi[:,i] <= c+d,
                            d[i] <= G[i].dot(D)*omega[:,i],
                            R*omega[:,i] <= r]
        problem = cvx.Problem(cost, constraints)
        optimal_value = problem.solve(solver=cvx.GUROBI, verbose=verbose)
        if optimal_value == np.inf:
            raiseError("One-shot LP unbounded")
        g = c.value+d.value
    else:
        # Apply the iterative method mentioned after Remark 3 in [1]
        
        def support(z,M,P,p):
            """
            Evaluate the support function for the set X={x : P*x<=p}:
            
                sigma(z | M*X) = sup{z^T*M*x : x \in {x : P*x <= p}}.
                
            Parameters
            ----------
            z : (n) array
                The direction along which to evalaute the support function.
            M : (n,m) array
                Direct mapping matrix for the set X={x : P*x<=p}.
            P : (n_p,m) array
                Template of the polytopic set X={x : P*x<=p}.
            p : (n_p) array
                Facet distances of the polytopic set X={x : P*x<=p}.
                
            Returns
            -------
            sigma : float
                The support function value.
            """
            m = M.shape[1]
            x = cvx.Variable(m)
            cost = cvx.Maximize(z.T.dot(M)*x)
            constraints = [P*x <= p]
            problem = cvx.Problem(cost, constraints)
            sigma = problem.solve(verbose=False)
            if sigma == np.inf:
                raiseError("Support function LP unbounded")
            return sigma
        
        g = np.zeros((n_g))
        iter_count = 0
        while True:
            iter_count += 1
            if iter_count > max_iter:
                raiseError("Iterative method ran out of iterations")            
            gs = [support(G[i],A,G,g)+support(G[i],D,R,r) for i in range(n_g)]
            gs = np.array(gs)          
            print "%d: %.4e" % (iter_count, la.norm(g-gs,ord=np.inf))
            if la.norm(g-gs,ord=np.inf) < cv_tol:
                g = np.matrix(gs).T
                break
            g = gs
        
    X = polytope.Polytope(G,g)
    return X

class MatlabEngineInterface:
    def __init__(self,meng):
        self.meng = meng
        
    def mset(self,vars_matlab,vars_python):
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
            self.meng.workspace[var_matlab] = matlab.double(var_python.tolist())
        
    def mget(self,var_matlab):
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
        var_python = np.array(self.meng.eval("%s" % (var_matlab)))
        return var_python

    def meval(self,expr):
        """
        Evaluate expression given by expr in MATLAB.
        
        Parameters
        ----------
        expr : str
            MATLAB expression to evaluate (just as you would type in the MATLAB
            command prompt).
        """
        self.meng.eval(expr,nargout=0)
    
def maxCRPI(A,B,D,C,G,g,H,h,R,r,meng=None,max_iter=100,cv_tol=1e-3,visualize=None):
    """
    Compute the Maximal Controlled Robust Positively Invariant (maxCRPI) set
    for the specified system
    
        x+ = A*x+B*u+D*p
        y = C*x
    
    and constraints
    
        y \in Y={y : G*y <= g}, u \in U={u : H*u <= h}, p \in P={p : R*p <= r}.
        
    Uses a modification of the algorithm presented in Seciton 4 of [1].
    
    [1] Kvasnica et al., "Reachability Analysis and Control Synthesis for
    Uncertain Linear Systems in MPT", 2015.
    
    Parameters
    ----------
    A : (n,n) array
        System dynamics A matrix (zero-input dynamics).
    B : (n,m) array
        System dynamics B matrix (how input enters the system).
    D : (n,d) array
        System dynamics D matrix (how disturbance enters the system).
    C : (p,n) array
        Output selection matrix (which defines the quantities that we care to
        keep invariant).
    G : (n_g,n) array
        Template of the safe outputs polytope.
    g : (n_g) array
        Facet distances of the safe outputs polytope.
    H : (n_h,m) array
        Template of the admissible inputs polytope.
    h : (n_h) array
        Facet distances of the admissible inputs polytope.
    R : (n_r,d) array
        Template of the tolerated disturbances polytope.
    r : (n_r) array
        Facet distances of the tolerated disturbances polytope.
    meng : MatlabEngine, optional
        Existing MATLAB engine, to save time and not start a new one every time
        this function is called. If provided, **the existsing workspace is
        cleared**.
    max_iter : int, optional
        Maximum number of iterations for the iterative algorithm.
    cv_tol : float, optional
        Convergence tolerance for stopping criterion for preimage set equality
        with the target set.
    visualize : callable, optional
        A function which accepts X the current preimage set and, for example,
        plots it.
    
    Returns
    -------
    maxCRPIset : Polytope
        The maxCRPI polytope.
    """
    def raiseError(error_string):
        raise AssertionError("[maxCRPI] %s" % (error_string))
        
    if meng is None:
        meng = MatlabEngineInterface(matlab.engine.start_matlab())
    else:
        meng = MatlabEngineInterface(meng)
        meng.meval("clear")
        
    # Nullspace of C (output selection matrix) as {x : nullC_A*x <= nullC_b}
    # polytope
    nullC_A = np.vstack((C,-C))
    nullC_b = np.zeros((C.shape[0]*2))
    pinvC = la.pinv(C) # Pseudoinverse of C
    
    # Setup workspace variables
    meng.mset(['A','B','D','C'],[A,B,D,C])
    meng.mset(['G','g','H','h','R','r'],[G,g,H,h,R,r])
    meng.mset(['nullC_A','nullC_b','pinvC'],[nullC_A,nullC_b,pinvC])
    meng.meng.workspace["cv_tol"] = cv_tol
    
    # Setup MPT3 polytopes
    meng.meval("Y = Polyhedron(G,g)")
    meng.meval("U = Polyhedron(H,h)")
    meng.meval("P = Polyhedron(R,r)")
    meng.meval("nullC = Polyhedron(nullC_A,nullC_b)")
    
    meng.meval("Omega = Y")
    iter_count = 0
    while True:
        iter_count += 1
        if iter_count > max_iter:
            raiseError("Ran out of iterations")
        meng.meval("pre = ((Omega-(((C*D)*P)+((C*A)*nullC)))+((-C*B)*U))*(C*A*pinvC)")
        meng.meval("Omega_next = pre & Omega")
        meng.meval("Omega_next_noredundant = Omega_next.minHRep")
        meng.meval("Omega_next = Omega_next.minHRep")
        num_facets = int(meng.meng.eval("size(Omega_next.A,1)"))
        print "%d: %d" % (iter_count, num_facets)
        meng.meval("stop = Polyhedron(Omega_next.A,Omega_next.b+cv_tol).contains(Omega) && "
                   "Polyhedron(Omega.A,Omega.b+cv_tol).contains(Omega_next)")
        stop = meng.meng.eval("stop")
        if stop:
            break
        else:
            meng.meval("Omega = Omega_next")
            if visualize is not None:
                X = polytope.Polytope(meng.mget("Omega.A"),meng.mget("Omega.b"))
                visualize(X)
                plt.pause(0.01)
    
    maxCRPIset = polytope.Polytope(meng.mget("Omega.A"),meng.mget("Omega.b"))
    return maxCRPIset
