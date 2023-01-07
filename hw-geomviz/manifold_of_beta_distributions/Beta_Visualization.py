import geomstats.backend as gs
from geomstats.information_geometry.beta import BetaDistributions

import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean

import matplotlib
import matplotlib.pyplot as plt

SE2_GROUP = SpecialEuclidean(n=2, point_type="matrix")

beta = BetaDistributions()


class Beta:
    
    """ Class for the visualization of beta manifold
    
    Parameters
    ----------
    points : array-like, shape=[..., 2]
            Point representing a beta distribution.
    
    Returns
    -------
    Plots
    """
    def plot(self,ax,size):
        """ Draws the beta manifold

                by Yiliang Chen

                Parameters
                ----------
                size : array-like, shape=[..., 2]
                    Defines the range of the manifold to be shown

        """
        ax.set(xlim=(0, size[0]), ylim=(0, size[1]))



    def plot_rendering(self,ax,initial_point=[0,0],size=[10,10],sampling_period=1):
        """ Draws the beta manifold

                by Yiliang Chen

                Parameters
                ----------
                size : array-like, shape=[..., 2]
                    Defines the range of the samples to be shown

                sampling_period: float, >0
                    Defines the sampling period of the sampled data
        """
        sz = gs.array(size)
        if sz.size != 2:
            raise ValueError(
                "size should be a 1*2 array"
            )
        x = gs.linspace(initial_point[0], (initial_point[0]+size[0]-1)*sampling_period, size[0])
        y = gs.linspace(initial_point[1], (initial_point[1]+size[1]-1)*sampling_period, size[1])
        points = [[i,j] for i in x for j in y]
        points_x = [i[0] for i in points]
        points_y = [i[1] for i in points]
        ax.scatter(points_x,points_y)

    def plot_grid(self,ax,size,initial_point=[0,0],n_steps=100,n_points=10,step=1,**kwargs):
        """ Draws the grids of beta manifold

                by Yiliang Chen

                Parameters
                ----------
                size : array-like, shape=[..., 2]
                    Defines the range of the grids to be shown

                step : float, >0
                    the length of a step for the grid

        """
        sz = gs.array(size)
        if sz.size != 2:
            raise ValueError(
                "size should be a 1*2 array"
            )
        b = [(initial_point[0]+i*step) for i in range(size[0])]
        gF = [(initial_point[1]+i*step) for i in range(size[1])]

        t = gs.linspace(0, 1, n_points)
        for i in b:
            for j in gF:
                start = [i,j]
                end_h = [i+step,j]
                end_v = [i,j+step]
                if i < (size[0] + initial_point[0] - 1):
                    grid_h=beta.metric.geodesic(initial_point=start,
                                                end_point=end_h,
                                                n_steps=n_steps)
                    ax.plot(*gs.transpose(gs.array([grid_h(k) for k in t])))
                if j < (size[1] + initial_point[1] - 1):
                    grid_v=beta.metric.geodesic(initial_point=start,
                                                end_point=end_v,
                                                n_steps=n_steps)
                    ax.plot(*gs.transpose(gs.array([grid_v(k) for k in t])))


    def scatter(self,ax,points,**kwargs):
        """ Scatter plot of beta manifold
        
        by Sunpeng Duan
    
        Parameters
        ----------
        points : array-like, shape=[..., 2]
            Point representing a beta distribution.
        """
        points_x = [gs.to_numpy(point[0]) for point in points]
        points_y = [gs.to_numpy(point[1]) for point in points]
        ax.scatter(points_x,points_y,**kwargs)
        
    def plot_geodesic(self,
                      ax,
                      initial_point,
                      end_point = None,
                      initial_tangent_vec = None,
                      n_steps = 100,
                      n_points = 10,
                      **kwargs):
        """ geomdesic plot of beta manifold
        
        by Sunpeng Duan
    
        Parameters
        ----------
        points : array-like, shape=[..., 2]
            Point representing a beta distribution.
        """

        
        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Specify an end point or an initial tangent "
                "vector to define the geodesic."
            )
            
        
        t = gs.linspace(0, 1, n_points)
            
        if end_point is not None:
            geod = beta.metric.geodesic(initial_point=initial_point, 
                                        end_point=end_point,
                                        n_steps=n_steps)(t)
            self.scatter(ax=ax,points=geod,**kwargs)
            
        if initial_tangent_vec is not None:
            geod = beta.metric.geodesic(initial_point=initial_point,
                                        initial_tangent_vec=initial_tangent_vec,
                                        n_steps=n_steps)(t)
            self.scatter(ax=ax,points=geod,**kwargs)
        
    def plot_geodestic_ball(self,
                      ax,
                      initial_point,
                      tangent_vecs,
                      n_steps = 100,
                      n_points = 10,
                      **kwargs):
        """ geomdesic plot of beta manifold
        
        by Sunpeng Duan
    
        Parameters
        ----------
        points : array-like, shape=[..., 2]
            Point representing a beta distribution.
        """
        
        t = gs.linspace(0, 1, n_points)
        
        for j in range(len(tangent_vecs)):
            geod = beta.metric.geodesic(initial_point=initial_point, 
                                        initial_tangent_vec=tangent_vecs[j, :],
                                        n_steps = n_steps)
            ax.plot(*gs.transpose(gs.array([geod(k) for k in t])))