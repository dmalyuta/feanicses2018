"""
Controllers.

Author: Danylo Malyuta.
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from scipy.linalg import solve_discrete_are as dare

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