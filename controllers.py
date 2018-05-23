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
    def setup(self,A,B,C,D,G,g,H,h,R,r,verbose=True):
        # Solve optimization problem
        n,m = B.shape
        p = C.shape[0]
        N = cvx.Variable(g.size,g.size)
        M = cvx.Variable(g.size,r.size)
        L = cvx.Variable(g.size,p)
        Q = cvx.Variable(h.size,g.size)
        S = cvx.Variable(h.size,p)
        K = cvx.Variable(m,n)
        pinvC = la.pinv(C)
        s = cvx.Variable(h.size)
        
        cost = cvx.Minimize(0)
# =============================================================================
#         constraints = [N*g+M*r <= g, #+s,
#                N*G == G.dot(A)+G.dot(B)*K,
#                #L*C == G.dot(A)+G.dot(B)*K,
#                M*R == G.dot(D),
#     #                       Q*g <= h,
#     #                       Q*G == H*K*pinvC,
#     #                       S*C == H*K,
#     #                       Q>=0,
#                N>=0, M>=0]
# =============================================================================
        constraints = [N*g+M*r <= g,
                       N*G == G.dot(C).dot(A).dot(pinvC)+G.dot(C).dot(B)*K*pinvC,
                       L*C == G.dot(C).dot(A)+G.dot(C).dot(B)*K,
                       M*R == G.dot(C).dot(D),
                       Q*g <= h,
                       Q*G == H*K*pinvC,
                       S*C == H*K,
                       Q>=0,
                       N>=0, M>=0]
                       
        problem = cvx.Problem(cost, constraints)
        optimal_value = problem.solve(solver=cvx.GUROBI, verbose=verbose)
        if optimal_value == np.inf:
            # This most likely means that a linear control law given the input
            # constraints and uncertainty set specifications cannot render the
            # state constraint set {x : G*x<=g} invariant.
            raise AssertionError("LP infeasible")
        self.s = s.value
        self.K = np.asarray(K.value)
        
    def oneStep(self,x):
        return self.K.dot(x)