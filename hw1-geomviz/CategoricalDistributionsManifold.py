from cmath import nan
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
from geomstats.information_geometry.categorical import CategoricalDistributions, CategoricalMetric
class CategoricalDistributionsManifold:
    def __init__(self, dim):
        self.dim = dim
        self.points = []
        self.ax = None
        self.elev, self.azim = None, None
        self.metric = CategoricalMetric(dim = self.dim)
        self.dist = CategoricalDistributions(dim = self.dim)
    def plot(self):
        min_limit = 0
        max_limit = 1
        plt.figure(dpi = 100)
        self.set_axis(min_limit, max_limit)
        if self.dim == 3:
            self.set_view()
            # x = np.linspace(start = min_limit, stop = max_limit, num = 201, endpoint = True)
            # y = np.linspace(start = min_limit, stop = max_limit, num = 201, endpoint = True)
            # X,Y = np.meshgrid(x,y)
            # Z = - X - Y + 1    
            # Z[Z<0] = nan
            # self.ax.plot_surface(X, Y, Z, vmin = 0, vmax = 1)
            x = [0, 1, 0, 0]
            y = [0, 0, 1, 0]
            z = [0, 0, 0, 1]
            vertices = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            tupleList = list(zip(x, y, z))
            poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
            self.ax.add_collection3d(Poly3DCollection(poly3d, edgecolors='k', facecolors='w', linewidths=3, alpha=0.2))

        elif self.dim == 2:
            X = np.linspace(start = min_limit, stop = max_limit, num = 101, endpoint = True)
            Y = 1 - X
            # self.ax.fill_between(X, Y)
            self.ax.plot(X, Y)


    def set_view(self, elev = 30.0, azim = 20.0):
        if self.dim == 3:
            if self.ax is None:
                self.set_axis()
            self.elev, self.azim = elev, azim
            self.ax.view_init(elev, azim)


    def set_points(self, points):
        self.points = points


    def scatter(self, n_samples, **scatter_kwargs):
        self.set_points(self.dist.random_point(n_samples=n_samples))
        self.plot()
        if self.dim == 3:
            for point in self.points:
                self.ax.scatter(point[0], point[1], point[2], **scatter_kwargs)
        elif self.dim == 2: 
            for point in self.points:
                self.ax.scatter(point[0], point[1], **scatter_kwargs)
        self.clear_points()

    def plot_geodesic(self, initial_point, end_point = None, tangent_vector = None):
        self.plot()
        geodesic = self.metric.geodesic(initial_point=initial_point, end_point = end_point, initial_tangent_vec = tangent_vector)
        num_samples = 100
        if self.dim == 3:
            for i in range(num_samples):
                point = geodesic(i/num_samples)
                self.ax.scatter(point[0], point[1], point[2], color='blue', s = 2)
            self.ax.scatter(geodesic(0)[0], geodesic(0)[1], geodesic(0)[2], color='red', s = 30)
            if tangent_vector is not None:
                normalized_tangent_vector = tangent_vector/np.sum(np.power(tangent_vector, 2))
                self.ax.quiver(
                    initial_point[0],
                    initial_point[1],
                    initial_point[2],
                    normalized_tangent_vector[0],
                    normalized_tangent_vector[1],
                    normalized_tangent_vector[2],
                    color = 'red',
                    length = 0.1,
                    normalize = True
                )
        elif self.dim == 2:
            for i in range(num_samples):
                point = geodesic(i/num_samples)
                self.ax.scatter(point[0], point[1], color='blue', s = 2)
            self.ax.scatter(geodesic(0)[0], geodesic(0)[1], color='red', s = 30)
            if tangent_vector is not None:
                normalized_tangent_vector = tangent_vector/np.sum(np.power(tangent_vector, 2))
                self.ax.quiver(
                    initial_point[0],
                    initial_point[1],
                    normalized_tangent_vector[0],
                    normalized_tangent_vector[1],
                    color = 'red',
                    angles = 'xy',
                    scale_units = 'xy',
                    scale = 10,
                )

    def plot_grid(self):
        self.plot()
        points = [
        np.array([0.5,0,0.5]),
        np.array([0,0.5,0.5]),
        np.array([0.5,0.5,0]),
        np.array([0.25,0,0.75]),
        np.array([0,0.25,0.75]),
        np.array([0.75,0,0.25]),
        np.array([0,0.75,0.25]),
        ]

        num_samples = 100
        curves = [(0,1),(0,2),(1,2),(3,2),(4,2),(3,4),(5,2),(6,2),(5,6)]
        for curve in curves:
            geodesic = self.metric.geodesic(initial_point=points[curve[0]], end_point= points[curve[1]])
            for i in range(num_samples):
                point = geodesic(i/num_samples)
                self.ax.scatter(point[0], point[1], color='black', s = 1)

    def clear_points(self):
        self.points = []

    def set_axis(self, min_limit, max_limit):
        if self.dim == 3: 
            ax = plt.subplot(111, projection="3d")
            plt.setp(
                ax,
                xlim = (min_limit, max_limit),
                ylim = (min_limit, max_limit),
                zlim = (min_limit, max_limit),
                anchor = (0,0),
                xlabel = "X",
                ylabel = "Y",
                zlabel = "Z",
            )
            
        elif self.dim == 2:
            ax = plt.subplot(111)
            plt.setp(
                ax,
                xlim = (min_limit, max_limit),
                ylim = (min_limit, max_limit),
                xlabel = "X",
                ylabel = "Y",
                aspect = "equal")

        self.ax = ax
