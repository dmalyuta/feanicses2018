"""
Polytopic computations library.

Author: Danylo Malyuta
        Autonomous Controls Laboratory, University of Washington

Version: 0.1 (March 2018)
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.spatial as ss

import cvxpy as cvx

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Polytope:
    """
    Class for a Polytope in H-representation.
    {x \in R^n | Px<=p}
    
    TODOs:
        - regiondiff where both P and Q are P-collections
        - polycover method 1 (regiondiff-based algorithm)
        - projection on lower dimensional space (C. Jones paper)
        - envelope
        - convex hull (conversion V-rep to H-rep)
        - pontryagin difference
        - minkowski sum
        - constructor as set of vertices
        - Hausdorff distance to another polytope (with the measure specified as
          whatever norm - 1, 2 or np.inf via ord= argument)
    """
    P = np.empty((1,1))
    p = np.empty((1,1))
#    D = None # Scaling matrix
    n = 0 # Number of dimensions of polytope
    n_p = 0 # Number of constraints
    tol = 1e-15 # Tolerance for emptyness/lower dimensionality check
    
    def __init__(self, P=[], p=[], R=[]):
        """
        Initializes polytope in H-representation.
        """
        # Convert to matrices
        P = np.mat(P)
        p = np.mat(p)

        # Construct P, p from R if R is passed non-empty
        N = len(R)
        if N > 0:
            # R is a list of tuples of the form:
            #    [(x1_min, x1_max),...,(xN_min,xN_max)]
            # Handle exceptions
            if P.size != 0 or p.size != 0:
                raise AssertionError("Cannot pass non-empty P, p when"
                                     "passing R")
            
            # Construct P, p based on R
            P = np.mat(np.zeros((2*N,N)))
            p = np.mat(np.zeros((2*N,1)))
            for i in range(N):
                xi_min = R[i][0]
                xi_max = R[i][1]
                if xi_min > xi_max:
                    raise AssertionError("R: x%d_min > x%d_max" % (i,i))
                P[2*i,i] = 1
                p[2*i] = xi_max
                P[2*i+1,i] = -1
                p[2*i+1] = -xi_min
            
        # Handle exceptions
        if p.shape[1] != 1:
            raise AssertionError("p must be a column vector")
        if P.shape[0] != p.size:
            raise AssertionError("Number of rows of P must match size of p")
        
        self.P = P
        self.p = p
        self.n_p = P.shape[0]
        self.n = P.shape[1]
        
    def raiseError(self, err_str):
        raise AssertionError("[Polytope] %s" % (err_str))
        
    def __getFeasiblePoint(self):
        """
        Get *some* feasible point in the polytope by solving a simple
        feasibility problem.
        
        (NOT USED, getChebyshevBall() used instead)
        """
        z = cvx.Variable(self.n)
        cost = cvx.Minimize(0) # Feasibility problem
        constraints = [self.P*z <= self.p]
        pbm = cvx.Problem(cost, constraints)
        pbm.solve()
        feasible_point = np.mat(z.value)
        return feasible_point
        
    def vrep(self, return_as_matrix=False):
        """
        Gets the vertex-representation of the polytope.
        Returns the set of its vertices.
        """
        halfspaces = np.block([self.P, -self.p])
        R_c, x_c = self.getChebyshevBall()
        interior_point = np.squeeze(np.asarray(x_c))
        vertices = ss.HalfspaceIntersection(halfspaces,
                                            interior_point).intersections
        if return_as_matrix:
            return vertices
        # Convert the matrix into a list of vectors
        v_list = []
        for i in range(vertices.shape[0]):
            v_list.append(np.mat(vertices[i]).T)
        return v_list
        
    def plot(self, axis=-1, coords=[], label="", **kwargs):
        """
        Plots the polytope on the given axis.
        """
        if self.isLowerDimensional() or self.isEmpty():
            # Will not plot empty or lower-dimensional polytopes
            return

        num_coords = len(coords)
        if num_coords > 1 and num_coords <= 3:
            vertices = self.project(coords)    
            
        if num_coords == 1 or num_coords > 3:
            raise AssertionError("Cannot plot projection onto more than 3 "
                                 "dimensions or onto 1 dimension")
        if self.n > 3 and num_coords == 0:
            raise AssertionError("Cannot plot higher dimensional polytope than"
                                 "3D")
            
        no_axis = axis == -1
        if no_axis:
            # Axis not passed in, create a new figure
            fig = plt.figure()
            if self.n == 2 or num_coords == 2:
                axis = fig.add_subplot(111)
            if self.n == 3 or num_coords == 3:
                axis = fig.add_subplot(111, projection='3d')
           
        # TODO plotting for self.n == 1
        if self.n == 2 or num_coords == 2:
            # 2D plot
            if num_coords == 0:
                # The polytope is 2D and we plot it as such
                vertices = self.vrep(return_as_matrix=True)
            co = ss.ConvexHull(vertices)
            # Filled region
            axis.fill(vertices[co.vertices,0],
                      vertices[co.vertices,1],
                      label=label, **kwargs)
        if self.n == 3  or num_coords == 3:
            # 3D plot
            if num_coords == 0:
                # The polytope is 3D and we plot is as such
                vertices = self.vrep(return_as_matrix=True)
            co = ss.ConvexHull(vertices)
            tri = matplotlib.tri.Triangulation(vertices[:,0],
                                               vertices[:,1],
                                               triangles=co.simplices)
            axis.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2],
                              triangles=tri.triangles, **kwargs)
            
        if no_axis:
            plt.show()
            
    def minDistance(self, z, norm="inf", verbose=False):
        """
        Compute the minimum distance between a point z and the polytope, using
        the provided norm.
        
        Parameters
        ----------
        norm : {1,2,"inf"}
            The norm to minimize.
        verbose : bool, optional
            If ``True``, solver prints its output.
            
        Returns
        -------
        d : float
            The minimum distance between z and this polytope.
        """
        x = cvx.Variable(self.n)
        cost = cvx.Minimize(cvx.pnorm(z-x,norm))
        constraints = [self.P*x <= self.p]
        problem = cvx.Problem(cost, constraints)
        min_distance = problem.solve(verbose=verbose)
        if min_distance==np.inf:
            self.raiseError("Infeasible problem.")
        return min_distance

    def minrep(self):
        """
        Compute minimum representation of the polytope, based on [1].
        Returns the minimum representation.
        
        [1] Baotic, M., "Polytopic Computations in Constrained Optimal
        Control", Automatika, 2009.
        """
        if self.n_p == 1:
            return self # Nothing to do if there is only one constraint
            
        I = range(self.n_p)
        for i in range(self.n_p):
            if len(I) == 1: # Down to one constraint, obviously not redundant
                break
            I.remove(i)
            # Solve relaxed LP
            x = cvx.Variable(self.n)
            cost = cvx.Maximize(self.P[i]*x)
            constraints = [self.P[I]*x <= self.p[I],
                           self.P[i]*x <= self.p[i]+1]
            pbm = cvx.Problem(cost, constraints)
            f_star = pbm.solve()
            if f_star > self.p[i]:
                # Relaxing the constraint "changed" the polytope, so add it
                # back
                I.append(i)
        return Polytope(self.P[I], self.p[I])
        
    def getChebyshevBall(self):
        """
        Returns the center x_c and the radius R_c of the Chebyshev ball of this
        polytope.
        """
        x_c = cvx.Variable(self.n)
        R_c = cvx.Variable()
        cost = cvx.Maximize(R_c)
        constraints = []
        for i in range(self.n_p):
            constraints.append(
                         self.P[i]*x_c+la.norm(self.P[i])*R_c <= self.p[i])
        pbm = cvx.Problem(cost, constraints)
        R_c_star = pbm.solve()
        return R_c_star, x_c.value

    def isEmpty(self):
        """
        Returns True if the polytope is empty.
        """
        R_c, x_c = self.getChebyshevBall()
        return R_c<-self.tol
        
    def isLowerDimensional(self):
        """
        Returns True if the polytope is lower dimensional.
        """
        R_c, x_c = self.getChebyshevBall()
        return np.abs(R_c)<=self.tol
        
    def intersect(self, P2):
        """
        Intersect this polytope with another polytope, P2.
        """
        Q = P2.P
        q = P2.p
        PP = np.block([[self.P],[Q]])
        pp = np.block([[self.p],[q]])
        P_intersect_Q = Polytope(PP,pp).minrep()
        return P_intersect_Q
        
    def regiondiff(self, QQ):
        """
        Set difference between this polytope and a P-collection QQ.
        
        Input:        
            QQ : a list of Polytope objects.
            
        Output:
            R : a list of Polytope objects, a P-collection representing
                P\QQ.
        """
        R = []
        k = 0
        # Check if a polytope in P-collection QQ such that 
        N_Q = len(QQ)
        for i in range(N_Q):
            n_q_i = QQ[i].n_p
            if n_q_i == 0:
                return R
        # Skip Q polytopes in QQ which do not overlap with P
        while Polytope(np.block([[self.P],[QQ[k].P]]),
                       np.block([[self.p],[QQ[k].p]])).isEmpty():
            k += 1
            if k >= N_Q:
                return self
        # Find new intersecting region
        n_q_k = QQ[k].n_p
        PP = Polytope(self.P, self.p)
        for j in range(n_q_k):
            if not Polytope(np.block([[PP.P],[-QQ[k].P[j]]]),
                            np.block([[PP.p],[-QQ[k].p[j]]])).isEmpty():
                Ptilde = PP.intersect(Polytope(-QQ[k].P[j], -QQ[k].p[j]))
                if k < N_Q-1:
                    new_regions = Ptilde.regiondiff(QQ[k+1:])
                    # Must make a list to be able to merge with R
                    if not hasattr(new_regions, "__iter__"):
                        new_regions = [new_regions]
                    R += new_regions
                else:
                    R += [Ptilde]
            PP = PP.intersect(Polytope(QQ[k].P[j], QQ[k].p[j]))
        return R
        
    def setdiff(self, Q):
        """
        Set difference between this polytope and another polytope Q.
        """
        return self.regiondiff([Q])
        
    def intervalHull(self):
        """
        Interval hull of the polytope, i.e. the smallest hyperrectangle
        containing the polytope.
        """
        R_hull = []
        x = cvx.Variable(self.n)
        constraints = [self.P*x <= self.p]
        for i in range(self.n):
            cost = cvx.Minimize(x[i])
            pbm = cvx.Problem(cost, constraints)
            xi_min = pbm.solve()
            cost = cvx.Maximize(x[i])
            pbm = cvx.Problem(cost, constraints)
            xi_max = pbm.solve()
            R_hull.append((xi_min,xi_max))
        return Polytope(R=R_hull)
        
    def extend(self, Q):
        """
        Extends the dimension of the polytope by adding another set of
        dimensions that are themselves within polytope Q.
        """
        P = sla.block_diag(self.P, Q.P)
        p = np.block([[self.p],[Q.p]])
        return Polytope(P,p)
        
    def randomPoint(self, N=1):
        """
        Return N random points inside the polytope.
        Hit-and-run sampling idea based on [1,2].
        
        [1] Tim Seguine (https://mathoverflow.net/users/17546/tim-seguine),
            Uniformly Sampling from Convex Polytopes, URL
            (version: 2014-04-03): https://mathoverflow.net/q/162327
        [2] https://www.cc.gatech.edu/~vempala/acg/notes.pdf
        """
        if N < 1:
            raise AssertionError("N>=1 required")
        # A first "non random" feasible seed point
        R_c, x_c = self.getChebyshevBall()
        points = [x_c] # Will contain the points
        for i in range(N):
            # Uniformly randomly pick a direction, each component in [-1,1]
            while True:
                direction = np.random.rand(self.n,1)*2.-1.
                direction_norm = la.norm(direction)  
                if direction_norm < 1e-5:
                    # Very unlucky, the random direction vector is quasi-zero
                    # Try again!
                    continue
                direction /= direction_norm
                break
            # Find theta_mina and theta_max such that x[-1]+theta*direction
            # is contained in the polytope
            theta = cvx.Variable()
            cost = cvx.Maximize(theta)
            constraints = [self.P*(points[-1]+theta*direction) <= self.p]
            pbm = cvx.Problem(cost, constraints)
            theta_max = pbm.solve()
            cost = cvx.Minimize(theta)
            pbm = cvx.Problem(cost, constraints)
            theta_min = pbm.solve()
            # Sample a random theta in [theta_min, theta_max]
            theta = np.random.uniform(low=theta_min, high=theta_max)
            # Add the new point
            points.append(points[-1]+theta*direction)
        del points[0] # Delete the first "non-random" feasible point
        for i in range(len(points)):
            points[i] = np.asarray(points[i].T).flatten()
        if len(points) == 1:
            return points[0]
        return points
        
    def project(self, coords):
        """
        Project the polytope onto a subset of its coordinates, resulting in a
        lower dimensional polytope.

        Input:
            coords - list of coordinates to project onto, each value must be
                     in [0,n-1]. They become, in the order given, the basis
                     vectors for the lower dimensional space of the projected
                     polytope.
        
        Returns the polytope in V-rep (vertex representation).
        
        TODO return the polytope in H-rep once I have the convert from V-rep to
             H-rep implemented. Then make it a public method
        TODO more efficient implementation? (E.g. Colin Jones' paper? Fourier-
             Motzkin elimination?)
        """
        n_reduced = len(coords) # Reduced dimension that output polytope
                                # will have
        vertices = self.vrep()
        vertices_proj = []
        for v_k in vertices:
            v_k_proj = np.mat(np.zeros((n_reduced,1)))
            j = 0
            for i in coords:
                e_i = np.mat(np.zeros((self.n, 1)))
                e_i[i] = 1
                e_i_reduced = np.mat(np.zeros((n_reduced, 1)))
                e_i_reduced[j] = 1
                j += 1
                v_k_proj += (v_k.T*e_i)[0,0]*e_i_reduced
            vertices_proj.append(v_k_proj.T)
        vertices_proj = np.asarray(np.vstack(vertices_proj))
        return vertices_proj
        
    def boundingBox(self, verbose=False):
        """
        Find  the smallest volume hyperrectangle that contains the polytope.
        The hyperrectangle is expressed as {x : l <= x <= u}.
        
        Parameters
        ----------
        verbose : bool, optional
            If ``True``, optimization solver prints its output.
        
        Returns
        -------
        l : array
            Hyperrectangle lower bound.
        u : array
            Hyperrectangle upper bound.
        """
        # Find upper bound
        u = cvx.Variable(self.n)
        Y = cvx.Variable(self.n,self.n_p)
        cost = cvx.Minimize(sum(u))
        constraints = [Y*self.p <= u,
                       Y*self.P == np.eye(self.n),
                       Y >= 0]
        problem = cvx.Problem(cost, constraints)
        optimal_value = problem.solve(verbose=verbose)
        if optimal_value == np.inf:
            raise AssertionError("Infeasible problem for bounding box")
        u = np.array(u.value.T).flatten()
        # Find lower bound
        l = cvx.Variable(self.n)
        cost = cvx.Maximize(sum(l))
        constraints = [Y*self.p <= -l,
                       Y*self.P == -np.eye(self.n),
                       Y >= 0]
        problem = cvx.Problem(cost, constraints)
        optimal_value = problem.solve(verbose=verbose)
        if optimal_value == np.inf:
            raise AssertionError("Infeasible problem for bounding box")
        l = np.array(l.value.T).flatten()
        return l,u
        
#    def normalize(self):
#        """
#        Scale the polytope such that the variable varies between [-1,1] in each
#        dimension. Modifies this very polytope!
#        """
#        l,u = self.boundingBox()
#        d = np.maximum(np.abs(l),np.abs(u))
#        self.D = np.diag(d)
#        self.P = self.P.dot(self.D)
#        
#    def denormalize(self):
#        """
#        Scale the polytope back.
#        """
#        if self.D is not None:
#            self.P = self.P.dot(la.inv(self.D))