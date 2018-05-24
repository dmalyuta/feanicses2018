"""
Controllers.

Author: Danylo Malyuta.
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from scipy.linalg import solve_discrete_are as dare
import cvxpy as cvx

import polytope as poly

def computeScalingMatrix(P,p):
    """
    Computes a scaling matrix D such that D*xhat=x where x is the
    original variable and xhat is the scaled one, ranging in [-1,1].
    Done for the polytope {x : P*x<=p}.
    
    Parameters
    ----------
    P : (m,n) array
        Polytope facet normals.
    p : (m) array
        Polytope facet distances.
        
    Returns
    -------
    D : (n,n) array
        The scaling matrix such that D*xhat=x.
    """
    l,u = poly.Polytope(P=P,p=p).boundingBox()
    d = np.maximum(np.abs(l),np.abs(u))
    D = np.diag(d)
    return D

class Controller:
    def __init__(self,*args,**kwargs):
        self.setup(*args,**kwargs)

    def setup(self):
        raise NotImplementedError("Must implement the setup() method")
        
    def __call__(self,x):
        return self.oneStep(x)
    
    def oneStep(self,x):
        raise NotImplementedError("Must implement the oneStep() method")
        
class LQR(Controller):
    def setup(self,A,B,Q,R):
        P = dare(A,B,Q,R)
        self.K = -la.inv(R+B.T.dot(P).dot(B)).dot(B.T).dot(P).dot(A)
        
    def oneStep(self,x):
        return self.K.dot(x)
    
class OnlineController(Controller):
    def setup(self,A,B,D,G,g,H,h,R,r,verbose=True):
        # Scaling matrices
        self.D_u = computeScalingMatrix(H,np.mat(h).T)
        
        self.A = A
        self.B = B
        self.D = D
        self.G = G
        self.g = g
        self.H = H
        self.h = h
        self.R = R
        self.r = r
        self.verbose = verbose
        
    def oneStep(self,x):
        n,m = self.B.shape
        Y = cvx.Variable(self.g.size,self.r.size)
        u = cvx.Variable(m)
        relax = cvx.Variable(self.g.size)
        
        cost = cvx.Minimize(cvx.norm(u,2))
        constraints = [Y*self.r+self.G.dot(self.B).dot(self.D_u)*u+self.G.dot(self.A).dot(x)<=self.g,
                       Y*self.R == self.G.dot(self.D),
                       self.H.dot(self.D_u)*u <= self.h,
                       Y>=0]

        problem = cvx.Problem(cost, constraints)
        optimal_value = problem.solve(solver=cvx.GUROBI, verbose=self.verbose)
        if optimal_value == np.inf:
            # This most likely means that a linear control law given the input
            # constraints and uncertainty set specifications cannot render the
            # state constraint set {x : G*x<=g} invariant.
            raise AssertionError("LP infeasible")
        u = self.D_u.dot(np.array(u.value).flatten())
        return u