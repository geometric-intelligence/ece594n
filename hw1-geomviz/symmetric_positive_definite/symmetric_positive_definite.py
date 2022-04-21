from calendar import c
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import math
import mpl_toolkits.mplot3d.art3d as art3d
from geomstats.visualization import Ellipses
from matplotlib.tri import Triangulation
from matplotlib.animation import FuncAnimation

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class SymmetricPositiveDefiniteVizualization:
    
    def __init__(self, maxZ = 1):
        self.maxZ = float(maxZ)
        self.currZ = self.maxZ
        self.ax = None
        self.spdPointViz = Ellipses()
        self.spdManifold = SPDMatrices(2)

    def cuboid_data(self, o, size=(1,1,1)):
        X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
            [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
        X = np.array(X).astype(float)
        X -= 0.5
        for i in range(3):
            X[:,:,i] *= size[i]
        X += np.array(o)
        return X


    def plotCubeAt(self, positions,sizes=None,colors=None, **kwargs):
        if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
        if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
        g = []
        for p,s,c in zip(positions,sizes,colors):
            g.append(self.cuboid_data(p, size=s) )
        return Poly3DCollection(np.concatenate(g),  
                                facecolors=np.repeat(colors,6, axis=0), **kwargs)


    def plot(self,  n_angles = 80, n_radii = 40, currZ=None, hsv=False):
        """Plots the 3D cone.   

        Parameters
        ----------
        n_angles : int
            Number of angles in polar coordinates            
        n_radii : int
            Number of radii in polar coordinates     
        currZ: Ignored
            Scaling factor     
        hsv: bool 
            Adds smooth gradient representation to the cone when set to True
            
        Returns
        -------
            Figure plot
        """
        
        if currZ == None:
            self.currZ = self.maxZ
        else:
            self.currZ = currZ

        #Modified from: https://stackoverflow.com/questions/55298164/3d-plot-of-the-cone-using-matplotlib
        # An array of radii
        # Does not include radius r=0, this is to eliminate duplicate points
        radii = np.linspace(0.0, self.currZ, n_radii)

        # An array of angles
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

        # Repeat all angles for each radius
        angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

        # Convert polar (radii, angles) coords to cartesian (x, y) coords
        # (0, 0) is added here. There are no duplicate points in the (x, y) plane
        x = np.append(0, (radii*np.cos(angles)).flatten())
        y = np.append(0, (radii*np.sin(angles)).flatten())


        # # Pringle surface
        z = np.full_like(x, self.currZ)

        # print(x.shape, y.shape, angles.shape, radii.shape, z.shape)
        # # NOTE: This assumes that there is a nice projection of the surface into the x/y-plane!
        tri = Triangulation(x, y)
        triangle_vertices = np.array([np.array([[x[T[0]], y[T[0]], z[T[0]]],

                                                [x[T[1]], y[T[1]], z[T[1]]],

                                                [x[T[2]], y[T[2]], z[T[2]]]]) for T in tri.triangles])
        x2 = np.append(0, (radii*np.cos(angles)).flatten())
        y2 = np.append(0, (radii*np.sin(angles)).flatten())
        
        # Pringle surface
        z2 = np.sqrt(x**2+y**2)

        # NOTE: This assumes that there is a nice projection of the surface into the x/y-plane!
        tri2 = Triangulation(x2, y2)

        triangle_vertices2 = np.array([np.array([[x2[T[0]], y2[T[0]], z2[T[0]]],

                                                [x2[T[1]], y2[T[1]], z2[T[1]]],

                                                [x2[T[2]], y2[T[2]], z2[T[2]]]]) for T in tri2.triangles])

        triangle_vertices = np.concatenate([triangle_vertices, triangle_vertices2])
        midpoints = np.average(triangle_vertices, axis=1)
        
        if hsv:
            facecolors = [self.find_color_for_point(pt) for pt in midpoints]  # smooth gradient
        else:
            facecolors = "0.75"  # grey

        coll = Poly3DCollection(
            triangle_vertices, facecolors=facecolors, edgecolors=None,  alpha=0.7, zorder=-10)
        self.artist = coll
        self.fig = plt.figure()
   
        self.ax = self.fig.gca(projection='3d')
        self.ax.add_collection(coll)

        self.ax.set_xlim(-self.maxZ*1.25, self.maxZ*1.25)
        self.ax.set_ylim(-self.maxZ*1.25, self.maxZ*1.25)
        self.ax.set_zlim(-self.maxZ*.25, self.maxZ*1.25)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.elev = 26
        

    @staticmethod
    def xyz_to_spd(point):
        x,y,z = point
        #let a = z
        #let x = b
        #let y = c
        
        # [z+x, y]
        # [y, z-x]
        #z>0
        #z^2 > x^2 + y^2
        
        return np.array([[z+x, y],[y, z-x]])
    
    @staticmethod
    def spd_to_xyz(point):
        z = (point[0,0] + point[1,1])/2.0
        x = point[0,0]-z
        y = point[0,1]
        
        
        return (x,y,z)


    def find_color_for_point(self, point):
        
       """Convert the color from HSV coordinates to RGB coordinates.

       Parameters
       ----------
       point : tuple-like of size = 3

       Returns
       -------
       color: tuple-like of size = 3     
       """
        x, y, z = point

        #convert radians to degrees
        angle = np.arctan2(x, y)*180/np.pi

        #normalize degrees to [0, 360]
        if (angle < 0):
            angle = angle + 360
        
        hue = angle/360
        saturation = math.sqrt(x**2+y**2)/self.maxZ
        value = z/self.maxZ

        color = colorsys.hsv_to_rgb(hue, saturation, value)

        return color

    def plot_grid():
        pass
    
    def plot_rendering_top(self, n_radii, n_angles):
        # Does not include radius r=0, this is to eliminate duplicate points
        # z_plane = self.maxZ
        z_plane = self.currZ
        radii = np.linspace(z_plane, 0, n_radii, endpoint=False)

        # An array of angles
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

        # Repeat all angles for each radius
        angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)


        x = np.append(0, (radii*np.cos(angles)).flatten())

        y = np.append(0, (radii*np.sin(angles)).flatten())

        # print("X Shape {}, Y Shape {}".format(x.shape, y.shape))
        for x_tmp, y_tmp in zip(x,y):
            # print("X: {} Y:{}".format(x_tmp, y_tmp))
            sampled_xyz = (x_tmp, y_tmp, z_plane)
            sampled_spd = SymmetricPositiveDefiniteVizualization.xyz_to_spd(sampled_xyz)
            # print(sampled_spd)
            ellipse_x, ellipse_y = self.spdPointViz.compute_coordinates(sampled_spd)
            ellipse_z = np.full_like(ellipse_x, z_plane)
            self.ax.plot(ellipse_x/(n_radii*n_angles*0.25)+sampled_xyz[0], ellipse_y/(n_radii*n_angles*0.25)+sampled_xyz[1], sampled_xyz[2], alpha = 0.8, zorder=10, color = self.find_color_for_point(sampled_xyz))


    def plot_rendering(self, n_radii=5, n_angles=16):
        '''
        draws the manifold with regularly sampled data.
        '''
        
        self.ax.elev = 90

        #plot top surface points
        self.plot_rendering_top(n_radii, n_angles)

        
        

    def plot_tangent_space(self, point):
        x, y, z = point

        positions = np.array([[x, y,  z]])
        pc = self.plotCubeAt(positions, sizes=[(.1,.1,.1*.5)]*len(positions), edgecolor="k",  alpha=0.8, zorder=10)
        self.ax.add_collection3d(pc)
 
        

    def scatter():
        pass

    def plot_geodesic(self, startPoint, endPoint):
        """
        allows to visualise a (discretised) geodesic. Takes either point and tangent vec as parameters, or initial point and end point as parameters.
        """
       
        # sphere = Hypersphere(dim=2)
        # point = np.array([-0.65726771, -0.02678122, 0.7531812])
        # vector = np.array([1, 0, 0.8])
        # tangent_vector = sphere.to_tangent(vector, base_point=point)
        # result = sphere.metric.exp(tangent_vector, base_point=point)
        baseMatrix = np.array([[1,0],[0,1]])
        point = self.spdManifold.projection(baseMatrix)
        print(point)

        finalMatrix = np.array([[1+0.5, 0.5],[0.5, 1-0.5]])
        # vector = self.spdManifold.to_vector(finalMatrix)
        tangent_matrix = self.spdManifold.random_tangent_vec(base_point=point)
        print("Tangent Matrix")
        print(tangent_matrix)
        tangent_vector = SymmetricPositiveDefiniteVizualization.spd_to_xyz(tangent_matrix)
        tangent
        # point = np.array([0,0,1])
        # vector = np.array([0.5,0.5,0])
        # tangent_vector = self.spdManifold.to_tangent(vector, base_point=point)
        # result = self.spdManifold.metric.exp(tangent_vector, base_point=point)

        vectorBasePoint = self.spdManifold.to_vector(point)
        print(vectorBasePoint)
        vectorBasePoint = SymmetricPositiveDefiniteVizualization.spd_to_xyz(point)
        # plt.plot(vectorBasePoint, ax=self.ax, label="Point")
        self.ax.scatter3D(vectorBasePoint[0], vectorBasePoint[1], vectorBasePoint[2])
        self.ax.quiver(vectorBasePoint[0], vectorBasePoint[1], vectorBasePoint[2], tangent_vector[0], tangent_vector[1], tangent_vector[2])
        # arrow = self.ax.Arrow3D(point, vector=tangent_vector) 
        # arrow.draw(self.ax, color="C0", label="Tangent Vector")
        # plt.plot(result, ax=self.ax, s=100, alpha=0.8, label="Exp", color="black")
        


    def plot_vector_field():
        pass

if __name__=="__main__":

    from geomstats.geometry.hypersphere import Hypersphere


    import matplotlib.pyplot as plt

    import symmetric_positive_definite
    import geomstats.visualization as visualization
    from geomstats.geometry.spd_matrices import *

    from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
    # Affine-invariant Riemannian metric 

    
    # randomPoints = spdManifold.random_point(n_samples=25000)

    # plot()

    # ellipses = visualization.Ellipses()


    # ellipses.draw_points(points=randomPoints)
    # symmetric_positive_definite.plot5()

    # symmetric_positive_definite.plot_hsv(False)

    viz = SymmetricPositiveDefiniteVizualization(1)

    viz.plot()
    viz.plot_geodesic(0,0)
    plt.show()
    # viz.plot(hsv=True)
    # plt.show(block=False)
    # input("Press Enter to continue...")

    # viz.plot(hsv=False)
    # viz.plot_rendering()
    # plt.show(block=False)
    # input("Press Enter to continue...")

    # viz.plot(currZ=0.9, hsv=False)
    # viz.plot_rendering()
    # plt.show(block=False)
    # input("Press Enter to continue...")

    # viz.plot(currZ=0.8, hsv=False)
    # viz.plot_rendering()
    # plt.show(block=False)
    # input("Press Enter to continue...")

    # viz.plot(currZ=0.7, hsv=False)
    # viz.plot_rendering()
    # plt.show(block=False)
    # input("Press Enter to continue...")


    # viz.plot(currZ=1, hsv=False)
    # viz.plot_tangent_space((0,0,1))
    # plt.show(block=False)
    # input("Press Enter to continue...")
    
    # viz.plot(currZ=0.5)
    # viz.plot_rendering(None)
    # plt.show(block=False)
    # input("Press Enter to exit...")
    # viz.plot_tangent_space((0,0,1))
    

    # viz.plot_tangent_space((0.5,0.5,1))
    # viz.
    
    
    
    
