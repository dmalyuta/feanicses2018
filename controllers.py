"""
Controllers.

Author: Danylo Malyuta.
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from scipy.linalg import solve_discrete_are as dare
import cvxpy as cvx

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
    
        cost = cvx.Minimize(cvx.norm(u,2))
        constraints = [Y*self.r+self.G.dot(self.B)*u+self.G.dot(self.A).dot(x) <= self.g,
                       Y*self.R == self.G.dot(self.D),
#                       Q*g <= h+s,
#                       Q*G == H*K,
#                       Q>=0,
                       Y>=0]
                       
        problem = cvx.Problem(cost, constraints)
        optimal_value = problem.solve(solver=cvx.GUROBI, verbose=self.verbose)
        if optimal_value == np.inf:
            # This most likely means that a linear control law given the input
            # constraints and uncertainty set specifications cannot render the
            # state constraint set {x : G*x<=g} invariant.
            raise AssertionError("LP infeasible")
        u = np.array(u.value).flatten()
        return u