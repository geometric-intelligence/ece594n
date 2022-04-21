from calendar import c
from unittest import result
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import math
import mpl_toolkits.mplot3d.art3d as art3d
from geomstats.visualization import Ellipses
from geomstats.geometry.spd_matrices import *
from matplotlib.tri import Triangulation
from matplotlib.animation import FuncAnimation
from geomstats.geometry.spd_matrices import SPDMatrices

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class SymmetricPositiveDefiniteVizualization:
    """This class provides all the essential methods for the
       visualization of the manifold of the Symmetric Positive 
       Definite matrices.
       
       Parameters
       ----------
       maxZ: int
          The scaling factor of the manifold
       
       Attributes
       ----------
       currZ: int
              The scaling factor of the manifold
       
       spdPointViz: Class used to plot points on the manifold SPD(2).
                    Elements S of the manifold of 2D Symmetric Positive 
                    Definite matrices can be conveniently represented by ellipses.
       
       spdManifold: Class for the manifold of symmetric positive definite (SPD) matrices.
                    Takes as input n (int) – Integer representing the shape of the matrices: n x n
                    By default n is set to 2 to in order for the visualization to be feasible.
                    
       metric: Compute the affine-invariant exponential map.Compute the Riemannian exponential at point 
               base_point of tangent vector tangent_vec wrt the metric
               defined in inner_product. This gives a symmetric positive definite matrix
                    
       See Also
       --------
       
       geomstats.geometry.spd_matrices.SPDMatrices
            Class for the manifold of symmetric positive definite (SPD) matrices
            
       geomstats.visualization.Ellipses
            Class used to plot points on the manifold SPD(2)
            
       geomstats.geometry.spd_matrices.SPDMetricAffine
            Class for the affine-invariant metric on the SPD manifold.
        
        
       References
       ----------
       [1] Miolane, Nina, et al. "Geomstats: a Python package for Riemannian geometry 
           in machine learning." Journal of Machine Learning Research 21.223 (2020): 1-9.
       
    """
    
    
    def __init__(self, maxZ = 1):
        self.maxZ = float(maxZ)
        self.currZ = self.maxZ
        self.ax = None
        self.spdPointViz = Ellipses()
        self.spdManifold = SPDMatrices(2)
        self.metric = SPDMetricAffine(2)

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
        """Plots a cube relative to specific coordinates.   

        Parameters
        ----------
        positions : array-like, size [1,3]
            Cordinates of the specific point           
        sizes : list of tuples
            Size of the cube-shaped tangent space     
        colors: string, optional (default=None)
            Specifies the color of the cube     
        
            
        Returns
        -------
            Figure plot
            
        See also
        -------
        
        matplotlib.collections.PolyCollection


        """
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
        currZ: int, optional (default=None)
            Scaling factor     
        hsv: bool, optional (default=False) 
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
            facecolors = "0.9"  # grey

        coll = Poly3DCollection(
            triangle_vertices, facecolors=facecolors, edgecolors=None,  alpha=0.5, zorder=-10)
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
    def elms_to_xyz(point):
        elm0,elm1,elm2 = point
        z = (elm0+elm2)/2
        y = elm1
        x = elm0-z     
        return (x,y,z)

    @staticmethod
    def xyz_to_elms(point):
        x,y,z = point
        elm0 = z+x
        elm1 = y
        elm2 = z-x
        return (elm0, elm1, elm2)


    @staticmethod
    def xyz_to_spd(point):
        """Converts cartesian coordinates to coordinates on the manifold coordinate system
        Parameters
        ----------
        point : tuple-like of size = 3
        
        Returns
        -------
        array-like, shape [2,2]   
        """
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
    def spd_to_xyz(matrix):
        """Converts coordinates on the manifold coordinate system to cartesian coordinates 
        Parameters
        ----------
        point : tuple-like of size = 3
        
        Returns
        -------
        tuple-like of size = 3
        """
        z = (matrix[0,0] + matrix[1,1])/2.0
        x = matrix[0,0]-z
        y = matrix[0,1]
        
        
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

    def plot_grid(self):
        self.plot_geodesic(startPointXYZ = (0, 0, 0.5), endPointXYZ = (0, 0, 0.6), n_geodesic_samples = 30)
        self.plot_geodesic(startPointXYZ = (0, 0, 0.5), endPointXYZ = (0, 0.1, 0.5), n_geodesic_samples = 30)
        self.plot_geodesic(startPointXYZ = (0, 0, 0.5), endPointXYZ = (0.1,0,0.5), n_geodesic_samples = 30)
    
    def plot_rendering_top(self, n_radii, n_angles):
        """Plots the ellipses (representations of the SPD matrices) on the tp of the cone manifold

        Parameters
        ----------
        n_angles : int
            Number of angles in polar coordinates            
        n_radii : int
            Number of radii in polar coordinates  

        Returns
        -------
        Figure plot    
        """
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
        """Plots the tangent space of the SPD manifold given a specific set of coordinates on the manifold
        Parameters
        ----------
        point : tuple-like, size 3
            Coordinates of the point based on which the tangent space will be plotted             
          
        Returns
        -------
        Figure plot    
        """
        x, y, z = point

        positions = np.array([[x, y,  z]])
        pc = self.plotCubeAt(positions, sizes=[(.1,.1,.1*.5)]*len(positions), edgecolor="k",  alpha=0.8, zorder=10)
        self.ax.add_collection3d(pc)
 
        

    def scatter(self, n_samples=100):
        """Plots a point cloud according to the manifold

        Parameters
        ----------
        n_samples : int
            Number of samples to be scattered             

        Returns
        -------
        Figure plot    
        """
        list_of_samples=[]
        samples=self.spdManifold.random_point(n_samples=n_samples)
        for i in samples:
            transf_sample=list(spd_to_xyz(i))
            list_of_transf_samples.append(transf_sample)
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        list_of_transf_samples=np.array(list_of_transf_samples)


        xs = list_of_transf_samples[:,0]
        ys = list_of_transf_samples[:,1]
        zs = list_of_transf_samples[:,2]
        self.ax.scatter(xs, ys, zs, marker='o')
        


    def plot_exp(self, startPointXYZ =  (0,0,1), tangentVectorXYZ=(0.5,0.5,-0.25)):
        
        # tangent_matrix = self.spdManifold.random_tangent_vec(base_point=SymmetricPositiveDefiniteVizualization.xyz_to_spd(startPointXYZ))
        tangent_matrix = SymmetricPositiveDefiniteVizualization.xyz_to_spd(tangentVectorXYZ)

        print("Tangent Matrix")
        print(tangent_matrix)


        tangent_vector = SymmetricPositiveDefiniteVizualization.
        
        
        (tangent_matrix)
        self.ax.scatter3D(startPointXYZ[0], startPointXYZ[1], startPointXYZ[2], label="Start Point")
        self.ax.quiver(startPointXYZ[0], startPointXYZ[1], startPointXYZ[2], tangentVectorXYZ[0], tangentVectorXYZ[1], tangentVectorXYZ[2], label="Tangent Vector")
        

        # resultMatrix = self.spdManifold.metric.exp(tangent_matrix, base_point=SymmetricPositiveDefiniteVizualization.xyz_to_spd(startPointXYZ))
        resultMatrix = self.metric.exp(tangent_matrix, base_point=SymmetricPositiveDefiniteVizualization.xyz_to_spd(startPointXYZ))

        print("Result")
        print(resultMatrix)
        resultXYZ = SymmetricPositiveDefiniteVizualization.spd_to_xyz(resultMatrix)
        self.ax.scatter3D(resultXYZ[0],resultXYZ[1],resultXYZ[2], label="Result: Point")
        self.ax.legend()

    
    
    def plot_log(self, startPointXYZ = (0,0,1), endPointXYZ = (0.25,0.25,0.5)):
        tangent_matrix = self.metric.log(SymmetricPositiveDefiniteVizualization.xyz_to_spd(endPointXYZ), base_point=SymmetricPositiveDefiniteVizualization.xyz_to_spd(startPointXYZ))

        # print("Tangent Matrix")
        # print(tangent_matrix)

        tangent_vector = SymmetricPositiveDefiniteVizualization.spd_to_xyz(tangent_matrix)
        
        self.ax.scatter3D(startPointXYZ[0], startPointXYZ[1], startPointXYZ[2], label="Start Point")
        self.ax.scatter3D(endPointXYZ[0], endPointXYZ[1], endPointXYZ[2], label="End Point")
        self.ax.quiver(startPointXYZ[0], startPointXYZ[1], startPointXYZ[2], tangent_vector[0], tangent_vector[1], tangent_vector[2], label="Result: Tangent Vector")
        self.ax.legend()

    def plot_geodesic(self, startPointXYZ = (0,0,1), endPointXYZ = (0.25,0.25,0.5), n_geodesic_samples = 30):
        
        """Allows the visualization of a (discretised) geodesic. Takes either point and tangent vec as parameters, or initial point and end point as parameters.

        Parameters
        ----------
        startPointXYZ : tuple-like, size 3
            Initial point of the geodesic            
        endPointXYZ : tuple-like, size 3
            End point of the geodesic  
        n_geodesic_sample: int
            Number of samples for discretization
            
        Returns
        -------
        Figure plot   
        """  
        tangent_matrix = self.metric.log(SymmetricPositiveDefiniteVizualization.xyz_to_spd(endPointXYZ), base_point=SymmetricPositiveDefiniteVizualization.xyz_to_spd(startPointXYZ))

        # print("Tangent Matrix")
        # print(tangent_matrix)

        tangent_vector = SymmetricPositiveDefiniteVizualization.spd_to_xyz(tangent_matrix)
        
        self.ax.scatter3D(startPointXYZ[0], startPointXYZ[1], startPointXYZ[2], label="Start Point")
        self.ax.scatter3D(endPointXYZ[0], endPointXYZ[1], endPointXYZ[2], label="End Point")
        self.ax.quiver(startPointXYZ[0], startPointXYZ[1], startPointXYZ[2], tangent_vector[0], tangent_vector[1], tangent_vector[2], label="Tangent Vector")
        
        result = self.metric.geodesic(initial_tangent_vec=tangent_matrix, initial_point=SymmetricPositiveDefiniteVizualization.xyz_to_spd(startPointXYZ))
        
        points_on_geodesic_spd = result(np.linspace(0.0, 1.0, n_geodesic_samples))
        
        geodesicXYZ = np.zeros((n_geodesic_samples,3))
        pointColors = []
        for index, matrix in enumerate(points_on_geodesic_spd):
            geodesicXYZ[index,:] = SymmetricPositiveDefiniteVizualization.spd_to_xyz(matrix)
            pointColors.append(self.find_color_for_point(geodesicXYZ[index,:]))

        
        self.ax.scatter3D(geodesicXYZ[1:-1,0], geodesicXYZ[1:-1,1],geodesicXYZ[1:-1,2], alpha=1, edgecolors="black",  color=pointColors[1:-1], label="Discrete Geodesic", zorder=100)        
        self.ax.legend()

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

    viz.plot(currZ=0.3)
    viz.plot_grid()
    # viz.plot_geodesic()
    # viz.plot_tangent_space(point=(0,0,1))
    # viz.plot_exp()
    
    # viz.plot_exp()
    
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
    
    
    
    
