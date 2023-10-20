import geomstats
import math
import geomstats.backend as gs
from geomstats.geometry.spd_matrices import *
from geomstats.geometry.lie_group import *
from geomstats.datasets.utils import load_connectomes
from geomstats.visualization.spd_matrices import Ellipses
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npla
from geomstats.learning.geodesic_regression import GeodesicRegression
from geomstats.geometry.pre_shape import PreShapeSpace, KendallShapeMetric
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.lie_group import LieGroup
from geomstats.geometry.general_linear import GeneralLinear
from shape_project import *


def unsqueeze(mat):
    return gs.reshape(mat,(1,mat.shape[0],mat.shape[1]))

def plot_mats(noisy_mats,nrow,ncol,step):
    #figs,axs=plt.subplots(1, ncol, figsize=(20,9))
    figs,axs=plt.subplots(nrow, ncol, figsize=(20,9))
    for im_id,noisy_mat in enumerate(noisy_mats):
        axs[int((im_id)/ncol)][(im_id)%ncol].imshow(noisy_mat)
        axs[int((im_id)/ncol)][(im_id)%ncol].title.set_text(f'T={im_id}')
        #axs[(im_id)%ncol].imshow(noisy_mat)
        #axs[(im_id)%ncol].title.set_text(f'T={im_id}')

def plot_graphs(noisy_mats,nrow,ncol):
    figs,axs=plt.subplots(nrow, ncol, figsize=(20,15))
    for im_id,noisy_mat in enumerate(noisy_mats):
        nx_graph = nx.from_numpy_array(np.ceil(noisy_mat))
        degrees = [n for n in nx.degree_centrality(nx_graph).values()]
        nx.draw(nx_graph,pos=None,with_labels=False,node_color=degrees,ax=axs[int((im_id)/ncol)][(im_id)%ncol],node_size=200,width=.75)
        axs[int((im_id)/ncol)][(im_id)%ncol].title.set_text(f'T={im_id}')
        
    return figs

def denoise_mats(x_0,noisy_samples):
    for ind,noisy_mat in enumerate(noisy_samples):
        if ind == 0:
            denoising_mats=unsqueeze(npla.solve(noisy_mat,x_0))
        else:
            denoising_mats=gs.concatenate((denoising_mats,unsqueeze(npla.solve(noisy_mat,noisy_samples[ind-1]))),axis=0)
            #denoising_mats=gs.concatenate((denoising_mats,unsqueeze(npla.solve(noisy_mat,x_0))),axis=0)
    return denoising_mats

def plot_results(label,test,noisy,title):
    figs,axs=plt.subplots(1, 3, figsize=(11,9))
    axs[0].imshow(noisy)
    axs[0].title.set_text(f"{title} noisy matrix")
    axs[1].imshow(test)
    axs[1].title.set_text(f"{title} generated matrix")
    axs[2].imshow(label)
    axs[2].title.set_text(f"{title} label matrix")

def compute_sqr_dist(a, b, metric):
    """Compute the Bures-Wasserstein squared distance.
        
    Compute the Riemannian squared distance between all 
    combinations of healthy SPD matrices and 
    schizophrenic SPD matrices.

    Parameters
    ----------
    healthy_spd : array-like, shape=[..., n, n]
        Point.
    schiz_spd : array-like, shape=[..., n, n]
        Point.
    n : int
        Size of matrix.

    Returns
    -------
    sqrd_dist : array-like, shape=[...]
        Riemannian squared distance of all SPD combinations.
    """
    sqrd_dist = []
    for i in range(len(a[:])):
        for j in range(len(b[:])):
            sqrd_dist.append(metric.squared_dist(a[i], b[j]))
    return sqrd_dist

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return gs.linspace(start, end, timesteps)

'''
def forward_diffusion_sample(x_0, shape, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = np.random.standard_normal(shape)
    # mean + variance
    return sqrt_alphas_cumprod[t] * x_0 \
    + sqrt_one_minus_alphas_cumprod[t] * noise, noise

# Define beta schedule
T = 10
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = gs.cumprod(alphas, axis=0)
alphas_cumprod_prev = gs.pad(alphas_cumprod[:-1], (1, 0))
alphas_cumprod_prev[0] = 1
sqrt_recip_alphas = gs.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = gs.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = gs.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
'''
def plot_graphs_spatial(nx_graph, subgraph_id, title):
    """Plots spatial graphs.
    
    Plots graph network with title and appropriate subgraph.
    
    Parameters
        ----------
        nx_graph : graph object
            Graph object used for plotting.
        subgraph_id : int
            ID for subgraph plot.
        title : string
            Title of plot.
    """
    plt.subplot(subgraph_id)
    degrees = [n for n in nx.degree_centrality(nx_graph).values()]
    nx.draw(nx_graph,pos=None,with_labels=False,node_color=degrees)
    plt.title(title)