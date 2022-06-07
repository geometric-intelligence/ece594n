"""Principal Curve Analysis on Manifolds.

Author: Siyu (Steven) Lin
"""
from geomstats.learning.frechet_mean import FrechetMean
import geomstats.backend as gs
import numpy as np
from scipy.linalg import svd
import torch

class PGA_Curve():
    r"""Principal curve analysis.
    Represent the data as a sequence of points joined by geodesics.
    
    Attributes:
    -----------
    metric : RiemannianMetric
        Riemannian metric.

    References:
    -----------
    Hauberg, S. (2015). Principal curves on Riemannian manifolds. IEEE transactions on pattern analysis and machine intelligence, 38(9), 1915-1921.
    """
    def __init__(self, metric):
        """ Construct a PGA_Curve object.

        Construct a PGA_Curve with a given Euclidean metric.

        Parameters:
        -----------
        metric : Euclidean Metric
            Metric of the Manifold.
        
        Returns:
        --------
        None.
        """
        self.metric = metric

    def fit(self, points, curve, num_iterations, sigma):
        """ Fit a principle curve.

        Fit a principle curve based on a given initial guess to the provided data points/

        Parameters:
        -----------
        points : array-like
            Array of data points to be fit.

        curve: array-like
            Array of points along a curve as the initial guess.
        
        num_iterations: integer
            The number of iterations to run the algorithm for convergence.

        sigma: float
            Measure of the spread of the smoothing kernel

        Returns:
        --------
        Array of points specifying the principle curve.
        """

        len_points = len(points)
        len_curve = len(curve)
        

        # Loop until converge
        for _ in range(num_iterations):
            # For each each point, find the closest point on the curve

            # dist[i][j] stores the distance from the i-th data point to the j-th point along the curve
            dist = np.zeros(shape = (len_points, len_curve))
            for i in range(len_points):
                for j in range(len_curve):
                    dist[i][j] = self.metric.dist(points[i], curve[j])

            # t_min[i] stores the index of the point along the curve that has the minimal distance to the i-th data point
            t_min = np.argmin(dist, axis = 1)
            # print(t_min)
    

            # Find the weights for each pair of data point and curve point

            w = np.zeros(shape = (len_curve, len_points))
            for i in range(len_curve):
                for j in range(len_points):
                    delta = self.metric.dist(curve[i], curve[t_min[j]]) / sigma
                    if np.abs(delta) <= 1: # Note: Finite support kernel
                        w[i][j] = (1-delta**2)**2 
            
            # print(w)

            # Update the estimate of the curve while keeping the self-consistency property:
            # Each point of the curve is the weighted Frechet mean of all the points 
            # print(points)
            for i in range(len_curve):
                mean_estimator = FrechetMean(self.metric)
                # mean_estimator.fit(points, weights = torch.tensor(w[i]).float())
                mean_estimator.fit(points, weights = gs.array(w[i]))


                curve[i] = mean_estimator.estimate_

        return curve
                



            





        
        
        

        




