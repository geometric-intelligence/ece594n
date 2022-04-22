import numpy as np
import matplotlib.pyplot as plt
import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_euclidean import _SpecialEuclideanVectors
import geomstats.backend as gs

# define manifold object and Reimannian metric
SE3_GROUP = SpecialEuclidean(n=3, point_type="vector")
METRIC = SE3_GROUP.left_canonical_metric

# implementation for SE(3)
class Special_Euclidean:
    
    def random_point(self, n_samples=1):
        """
        Produces a random point of a 3D Special Euclidean manifold.
        
        Parameters
        ------- 
        n_samples : int, optional
            Number of random_point items to create.
        """
        random_point = SE3_GROUP.random_point(n_samples=n_samples)
        return random_point
        
    def plot(self, point):
        """
        Plots a 6D point in the Special Euclidean manifold onto a 3D grid. 
        
        Parameters
        ------- 
        points : array-like, shape=[..., dim]
            Points to be plotted.
        """
        print(f"Point: {point}") # point
        print(f"Rotation: {point[:3]}") # rotation
        print(f"Translation: {point[3:]}") # translation

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        visualization.plot(point, ax=ax, space="SE3_GROUP")
        
    def scatter(self, num_points):
        """
        Simulates a point cloud by plotting multiple 6D points in the Special Euclidean manifold onto a 3D grid. 
        
        Parameters
        ------- 
        num_points : int
            Number of points in the point cloud.
        """
        random_point = SE3_GROUP.random_point(n_samples=num_points)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        visualization.plot(random_point, ax=ax, space="SE3_GROUP")
        
    def plot_exp(self, initial_point, end_point):
        """
        Plots the visualization of the exponential function (from the Geomstats package).  
        
        Parameters
        ------- 
        intial_point : array-like, shape=[..., dim]
            Initial point for the exponential function.
        final_point : array-like, shape=[..., dim]
            Final point for the exponential function.
        """
        # remove duplicates from the legend
        def legend_without_duplicate_labels(ax):
            handles, labels = ax.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            ax.legend(*zip(*unique))
    
        #take the exp of two points and plot
    
        #set figure size
        plt.rcParams['figure.figsize'] = [10, 10]

        #create two random points
        initial_point = initial_point
        end_point = end_point

        # create a array of points
        points = np.append(end_point, initial_point)

        #compute the exp, between initial and end point
        exp = METRIC.exp(end_point, initial_point)

        #print the points
        print('the initial point is:        ', initial_point,'\n')
        print('the end point is:            ',end_point,'\n')
        print('the exp of the two points is:',exp,'\n')

        #create figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        #plot the initial, end and exp
        ax = visualization.plot(initial_point, ax=ax, space="SE3_GROUP",color='green',label = 'initial')
        ax = visualization.plot(exp,ax=ax, space = "SE3_GROUP",color='purple',label='exp')
        ax = visualization.plot(end_point, ax=ax, space="SE3_GROUP",color = 'red',label = 'end')

        #plot a line between the points
        x_values = [initial_point[3],end_point[3]]
        y_values = [initial_point[4],end_point[4]]
        z_values = [initial_point[5],end_point[5]]

        plt.plot(x_values,y_values,z_values,color='blue',linestyle = 'dashed')


        #set legend size and location
        legend_without_duplicate_labels(ax)

        #set graph limits
        a=2
        plt.xlim((-a,a))
        plt.ylim((-a,a))
        ax.set_zlim(-a,a)

        #show plot
        plt.show()
        
    def plot_log(self, initial_point, end_point):
        """
        Plots the visualization of the logarithmic function (from the Geomstats package).
        With Geodesic transition between points by taking the log of the initial and end points.   
        
        Parameters
        ------- 
        intial_point : array-like, shape=[..., dim]
            Initial point for the logarithmic function.
        final_point : array-like, shape=[..., dim]
            Final point for the logarithmic function.
        """
        # remove duplicates from the legend
        def legend_without_duplicate_labels(ax):
            handles, labels = ax.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            ax.legend(*zip(*unique))

        #take the log of two points and plot

        #set figure size
        plt.rcParams['figure.figsize'] = [10, 10]

        #create two random points
        end_point = end_point
        initial_point = initial_point

        # create a array of points
        points = np.append(end_point, initial_point)

        #compute the log, between initial and end point
        log = METRIC.log(end_point, initial_point)

        #print the points
        print('the initial point is:        ', initial_point,'\n')
        print('the end point is:            ',end_point,'\n')
        print('the log of the two points is:',log,'\n')

        #create figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        #plot the initial, end and log
        ax = visualization.plot(initial_point, ax=ax, space="SE3_GROUP",color='green',label = 'initial')
        ax = visualization.plot(log,ax=ax, space = "SE3_GROUP",color='purple',label='log')
        ax = visualization.plot(end_point, ax=ax, space="SE3_GROUP",color = 'red',label = 'end')

        #plot a line between the points
        x_values = [initial_point[3],end_point[3]]
        y_values = [initial_point[4],end_point[4]]
        z_values = [initial_point[5],end_point[5]]

        plt.plot(x_values,y_values,z_values,color='blue',linestyle = 'dashed')


        # COMPUTE GEODESIC:
        # init_tangent_vec is log

        geodesic = METRIC.geodesic(
            initial_point=initial_point, initial_tangent_vec=log
        )

        #generate time for geodesic
        N_STEPS = 2
        t = gs.linspace(0.3, .6, N_STEPS)

        #generate the points
        points = geodesic(t)

        visualization.plot(points, space="SE3_GROUP")


        #set legend size and location
        legend_without_duplicate_labels(ax)

        #set graph limits
        a=2
        plt.xlim((-a,a))
        plt.ylim((-a,a))
        ax.set_zlim(-a,a)

        #show plot
        plt.show()
        
        
    def plot_geodesic(self, point, vector, N_STEPS):
        """
        Plots a geodesic of SE(3).
        SE3 is equipped with its left-invariant canonical metric.
        
        Parameters
        ------- 
        point : array-like, shape=[..., dim]
            Point for the geodesic function.
        vector : array-like, shape=[..., dim]
            Vector for the geodesic function.
        N_STEPS : array-like, shape=[..., dim]
            Number of samples on the geodesic to plot.
        """
        N_STEPS = N_STEPS

        # passes in a point and vector to the geodesic function of the left canonical matrix type
        initial_point = point
        initial_tangent_vec = gs.array(vector)
        geodesic = METRIC.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )

        # defines the sampling of points on the geodesic
        t = gs.linspace(-3.0, 3.0, N_STEPS)

        points = geodesic(t)
        
        # creates figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        visualization.plot(points, ax=ax, space="SE3_GROUP")
        
        a=3
        plt.xlim((-a,a))
        plt.ylim((-a,a))
        ax.set_zlim(-a,a)
        
        plt.show()