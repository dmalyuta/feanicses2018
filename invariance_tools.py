"""
Functions for generating invariant sets.

Author: Danylo Malyuta.
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import cvxpy as cvx

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

def trodden(A,D,R,r,G=None,iterative=False,cv_tol=1e-8,max_iter=np.inf,
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
        raise AssertionError("[trodden] %s" % (error_string))
    
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