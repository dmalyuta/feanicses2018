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
    
class LFS(Controller):
    def setup(self,A,B,D,G,g,H,h,R,r,verbose=True):
        # Solve optimization problem
        g = np.copy(g)
        iter_count = 0
        while True:
            iter_count += 1
            if iter_count > 1:
                q_prev = q.value
            n,m = B.shape
            N = cvx.Variable(g.size,g.size)
            M = cvx.Variable(g.size,r.size)
            Q = cvx.Variable(h.size,g.size)
            K = cvx.Variable(m,n)
            q = cvx.Variable(g.size)
            s = cvx.Variable(h.size)
        
            cost = cvx.Minimize(cvx.norm(q,'inf')+cvx.norm(s,'inf'))
            constraints = [N*g+M*r <= g+q,
                           N*G == G.dot(A)+G.dot(B)*K,
                           M*R == G.dot(D),
                           Q*g <= h+s,
                           Q*G == H*K,
                           Q>=0,
                           N>=0, M>=0]
                           
            problem = cvx.Problem(cost, constraints)
            optimal_value = problem.solve(solver=cvx.GUROBI, verbose=False) #verbose)
            if optimal_value == np.inf:
                # This most likely means that a linear control law given the input
                # constraints and uncertainty set specifications cannot render the
                # state constraint set {x : G*x<=g} invariant.
                raise AssertionError("LP infeasible")
               
            print la.norm(q.value,ord=np.inf)
            if la.norm(q.value,ord=np.inf) < 1e-5:
                break
            else:
                g += q.value
        self.s = s.value
        self.q = q.value
        self.K = np.asarray(K.value)
        
    def oneStep(self,x):
        return self.K.dot(x)