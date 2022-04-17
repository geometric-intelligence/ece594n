import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import colorsys

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
    symmetric_positive_definite.plot5()


