import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import math

from matplotlib.tri import Triangulation

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# def plot(points):
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     xdata = points[:,0,0]
#     ydata = points[:,0,1]
#     zdata = points[:,1,1] 
#     ax.scatter3D(xdata, ydata, zdata, c=points[:,1,1], cmap='viridis',linewidth=0.5);
#     # fig.show()
#     plt.show()

# def plot2(points):
#     ax = plt.axes(projection='3d')

#     xdata = points[:,0,0]
#     ydata = points[:,0,1]
#     zdata = points[:,1,1] 
#     ax.plot_surface(xdata, ydata, zdata, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='blue')

#     ax.set_title('surface'); 


# def plot4():
#     #should be odd
#     ZSIZE=101
#     z = np.outer(np.linspace(0, 1, ZSIZE), np.ones(ZSIZE))

#     x = np.zeros((ZSIZE,ZSIZE))
#     midpoint = int(ZSIZE/2)
#     for i in range(ZSIZE):
#         currZ = z[i,0]
        
#         for j in range(midpoint):
#             currMag = currZ*2*(j+1)/ZSIZE
#             x[i,j]= currMag
#             x[i,j+midpoint]= -currMag

   
#     y= np.sqrt(z**2-x**2) -np.sqrt(z**2-x**2)
#     print(x)
#     print(y)
#     print(z)

#     print(x.shape)
#     print(y.shape)
#     print(z.shape)

#     fig = plt.figure()
 
#     # syntax for 3-D plotting
#     ax = plt.axes(projection ='3d')

#     # syntax for plotting
#     ax.plot_surface(x, y, z, cmap ='viridis', edgecolor ='green')
#     x=-x
#     y= -y.copy()
#     ax.plot_surface(x, y, z, cmap ='viridis', edgecolor ='green')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z');
#     # ax.set_title('Surface plot geeks for geeks')
#     plt.show()
    

# def f(x, y):
#     return np.sqrt(x ** 2 + y ** 2)

# def fz(z):
#     pass
#     # return np.

# def plot3():
#     x = np.linspace(-1, 1, 100)
#     y = np.linspace(-1, 1, 100)
#     # z = np.linspace(0,1,100)
    
#     # Z = np.meshgrid(z)
#     # X,Y = fz(z)
#     X, Y = np.meshgrid(x, y)
#     Z = f(X, Y)

#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.contour3D(X, Y, Z, 50, cmap='binary', edgecolor='blue')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z');
#     plt.show()

# def hsv(t, r, z): 
#     #let z be value

#     #let radius be saturation

#     #let hue be angle


#     return (np.cos(x + 2 * y) + 1) / 2

def plot_hsv():

    #Modified from: https://stackoverflow.com/questions/55298164/3d-plot-of-the-cone-using-matplotlib
    n_angles = 80

    n_radii = 40


    # An array of radii

    # Does not include radius r=0, this is to eliminate duplicate points

    radii = np.linspace(0.0, 1.0, n_radii)


    # An array of angles

    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)


    # Repeat all angles for each radius

    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)


    # Convert polar (radii, angles) coords to cartesian (x, y) coords

    # (0, 0) is added here. There are no duplicate points in the (x, y) plane


    x = np.append(0, (radii*np.cos(angles)).flatten())

    y = np.append(0, (radii*np.sin(angles)).flatten())


    # # Pringle surface

    # z = 1+-np.sqrt(x**2+y**2)*2
    z = np.full_like(x, 1)

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

    facecolors = [find_color_for_point(pt) for pt in midpoints]  # smooth gradient
    # facecolors = [np.random.random(3) for pt in midpoints]  # random colors


    coll = Poly3DCollection(
        triangle_vertices, facecolors=facecolors, edgecolors=None)


    fig = plt.figure()

    ax = fig.gca(projection='3d')

    ax.add_collection(coll)

    ax.set_xlim(-1, 1)

    ax.set_ylim(-1, 1)

    ax.set_zlim(0, 1)

    ax.elev = 25

    # rotate the axes and update
    for angle in range(0, 36):
        ax.view_init(25, angle*10)
        plt.draw()
        plt.pause(.001)

    # plt.show()


# def find_color_for_point(pt):

#     c_x, c_y, c_z = pt

#     angle = np.arctan2(c_x, c_y)*180/np.pi

#     if (angle < 0):
#         angle = angle + 360

#     if c_z < 0:

#         l = 0.5 - abs(c_z)/2
#         #l=0
#     if c_z == 0:
#         l = 0.5
#     if c_z > 0:
#         l = (1 - (1-c_z)/2)

#     if c_z > 0.97:

#         l = (1 - (1-c_z)/2)

#     col = colorsys.hls_to_rgb(angle/360, l, 1)

#     return col

def find_color_for_point(pt):

    c_x, c_y, c_z = pt

    angle = np.arctan2(c_x, c_y)*180/np.pi

    if (angle < 0):
        angle = angle + 360

    hue = angle/360
    saturation = math.sqrt(c_x**2+c_y**2)
    value = c_z
    color = colorsys.hsv_to_rgb(hue, saturation, value)

    return color




def plot5():
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    # Set up the grid in polar
    theta = np.linspace(0,2*np.pi,90)
    r = np.linspace(0,1,50)
    T, R = np.meshgrid(theta, r)

    


    # Then calculate X, Y, and Z
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Z = np.sqrt(X**2 + Y**2)

    # Set the Z values outside your range to NaNs so they aren't plotted
    Z[Z < 0] = np.nan
    Z[Z > 1] = np.nan
    ax.plot_surface(X, Y, Z, cmap="plasma")



    ax.set_zlim(0,1)

    plt.show()

def plot_grid():
    pass

def plot_rendering():
    pass

def plot_tangent_space():
    pass

def scatter():
    pass

def plot_geodesic():
    pass

def plot_vector_field():
    pass

if __name__=="__main__":
    import matplotlib.pyplot as plt

    import symmetric_positive_definite
    import geomstats.visualization as visualization
    from geomstats.geometry.spd_matrices import *

    spdManifold = SPDMatrices(2)
    # randomPoints = spdManifold.random_point(n_samples=25000)

    # plot()

    # ellipses = visualization.Ellipses()

    # ellipses.draw_points(points=randomPoints)
    # symmetric_positive_definite.plot5()

    symmetric_positive_definite.plot_hsv()

