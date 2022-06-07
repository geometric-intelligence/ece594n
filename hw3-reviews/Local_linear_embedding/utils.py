
"""

The following functions implement :
[A. Goh and R. Vidal, "Clustering and dimensionality reduction on Riemannian manifolds," 2008 IEEE Conference on Computer Vision and Pattern Recognition, 2008, pp. 1-7, doi: 10.1109/CVPR.2008.4587422.](https://ieeexplore.ieee.org/document/)
Author : Abhijith Atreya

"""

import numpy as np
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import eigsh
from geomstats.geometry.pullback_metric import PullbackMetric
from geomstats.learning.knn import KNearestNeighborsClassifier
from geomstats.geometry.euclidean import Euclidean
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import ticker

np.random.seed(10)

def Compute_neighbours(data, labels, metric, n_neighbors):
    """
    Compute nearest k-nearest neighbours according to the distance metric specified 
    Parameters
    ----------
    data : array-like, shape (n_samples, n_dim)
    labels : array-like, shape (n_samples, )
    metric : distance metric to be used
    n_neighbors : number of nearest neighbours to compute
    Returns
    -------
    k_nearest_vals : array-like, shape (n_samples, n_neighbors)
    """
    knn = KNearestNeighborsClassifier(n_neighbors= n_neighbors, distance=metric.dist)
    knn.fit(data,labels)
    k_nearest_vals = knn.kneighbors(data, return_distance=False)
    print("K-nearest values of each point:")
    print(k_nearest_vals[:3,:])
    return k_nearest_vals

def barycenter_weights(metric, X, Y, indices, reg=1e-3):
    """
    Compute barycenter weights of X from Y along the first axis
    We estimate the weights to assign to each point in Y[indices] to recover
    the point X[i]. The barycenter weights sum to 1.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_dim)
    Y : array-like, shape (n_samples, n_dim)
    indices : array-like, shape (n_samples, n_dim)
            Indices of the points in Y used to compute the barycenter
    reg : float, default=1e-3
        amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim
    Returns
    -------
    B : array-like, shape (n_samples, n_neighbors)
    Notes
    -----
    This code implements the manifold version of 
    "https://github.com/scikit-learn/scikit-learn/blob/80598905e517759b4696c74ecc35c6e2eb508cff/sklearn/manifold/_locally_linear.py#L122"
    """

    n_samples, n_neighbors = indices.shape
    assert X.shape[0] == n_samples

    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    for i, ind in enumerate(indices):
        A = Y[ind]
        C = metric.log(A,X[i])
        G = np.dot(C,C.T)
        #G = metric.norm(G)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[:: n_neighbors + 1] += R
        w = solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
        
    return B

def Compute_W(data, metric ,k_nearest_vals, n_samples, n_neighbors ):
    """
    Compute the weight matrix : W (2) in "Clustering and Dimensionality Reduction on Riemannian Manifolds"
    Parameters
    ----------
    data : array-like, shape (n_samples, n_dim)
    metric : distance metric to be used
    k_nearest_vals : array-like, shape (n_samples, n_neighbors)
    n_samples : int
    n_neighbors : number of nearest neighbours to compute
    Returns
    -------
    M,W : array-like, shape (n_samples, n_samples)
    """
    B= barycenter_weights(metric, data,data,k_nearest_vals);
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    W = csr_matrix((B.ravel(), k_nearest_vals.ravel(), indptr), shape=(n_samples, n_samples))
    W_mat = W.toarray()
    M = eye(*W.shape, format=W.format) - W
    M = (M.T * M).toarray()
    return M , W

def Validate_W(data, W, k_nearest_vals):
    """
    Validate the weight matrix : W (2) in "Clustering and Dimensionality Reduction on Riemannian Manifolds"
    Plots the reconstructed data (in 2D) from the weighted neighbours against the actual values.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_dim)
    W : array-like, shape (n_samples, n_samples)
    k_nearest_vals : array-like, shape (n_samples, n_neighbors)
    """
    linear_combos = []
    neighborhood_weights = []
    for i in range(len(data)):
        weights = W.toarray()[i][k_nearest_vals[i]]
        neighborhood = data[k_nearest_vals[i]]
        weighted_neighbors = weights.reshape(-1,1)*neighborhood
        x1 = np.sum(weighted_neighbors[:,0])
        x2 = np.sum(weighted_neighbors[:,1])
        x3 = np.sum(weighted_neighbors[:,2])
        linear_combos.append([x1, x2, x3])
        neighborhood_weights.append(weights)
    reconstructed = np.array(linear_combos)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121)
    ax.scatter(reconstructed[:,0], reconstructed[:,2], c='red', s=50, label='Linear Reconstruction')
    ax.scatter(data[:,0], data[:,2], c='blue', s=10, label='Original Data')
    ax.set_title('Local Linear Combinations')
    ax.set_xlabel("Dimension 0")
    ax.set_ylabel("Dimension 2")
    ax.legend();
    ax2 = fig.add_subplot(122)
    ax2.scatter(reconstructed[:,0], reconstructed[:,1], c='red', s=50, label='Linear Reconstruction')
    ax2.scatter(data[:,0], data[:,1], c='blue', s=10, label='Original Data')
    ax2.set_title('Local Linear Combinations')
    ax2.set_xlabel("Dimension 0")
    ax2.set_ylabel("Dimension 1")
    ax2.legend();
    return

class Optimize_y():
    """
    This class computes the objective functio: 'y' (3) in 
    "Clustering and Dimensionality Reduction on Riemannian Manifolds"
    
    Parameters
    ----------
    M : array-like, shape (n_samples, n_samples)
    n_samples : int
    n_components : int
    """
    def __init__(self, M, n_samples,n_components) :
        self.n_dimensions = n_components
        self.n_samples = n_samples
        self.Y0 = np.random.rand(n_samples, self.n_dimensions)
        self.M = M

    def objective_function(self,Y):
        Y = np.reshape(Y,(self.n_samples, self.n_dimensions))
        return np.trace(np.matmul (np.matmul(Y.T , self.M) , Y))

    def cons1(self,y):
        y = np.reshape(y,(self.n_samples, self.n_dimensions))
        return np.sum(y ,axis = 1)

    def cons2(self,y):
        var = 0
        rotational_cons = np.zeros((np.shape(y)))
        y = np.reshape(y,(self.n_samples, self.n_dimensions))
        for i in range(self.n_samples):
            y_i = np.array(y[i,:])[np.newaxis]
            var1 = y_i.T @ y_i
            var = var + var1;
        rotational_cons = (1/ self.n_samples)*var - eye(2,2)
        aaa = np.squeeze(np.asarray(rotational_cons.reshape(4,1)))
        return aaa

    def mimimize(self):
        cons = ({'type': 'eq', 'fun': self.cons1},
                {'type': 'eq', 'fun': self.cons2}
                )
        a = self.objective_function(self.Y0)
        obj = minimize( self.objective_function , self.Y0 , constraints=cons)
        return obj

def null_space(M, k):
    """
    Compute the null space of M : (Sparse eigenvalue problem) in 
    "Clustering and Dimensionality Reduction on Riemannian Manifolds"
    Parameters
    ----------
    M : array-like, shape (n_samples, n_samples)
    k : int
    Returns
    -------
    eigen_vectors, eigen_values_sum
    """
    k_skip = 1
    tol = 1e-6
    max_iter = 100
    random_state = None
    #v0 = random_state.uniform(-1, 1, M.shape[0])
    eigen_values, eigen_vectors = eigsh(
                M, k + k_skip, sigma=0.0, tol=tol, maxiter=max_iter )
    return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])

def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(8, 8),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')   
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
