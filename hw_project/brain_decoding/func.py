from __future__ import print_function, division

# Basics:
import numpy as np,pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import os, random, time, sys, copy, math, pickle

plt.ion()

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from nibabel.testing import data_path
from nilearn import plotting as nplt
from nilearn.input_data import NiftiMasker
from nilearn import datasets
from nilearn import plotting
from nilearn.image import mean_img
from nilearn.image import index_img
import nibabel as nib
from nilearn import image

from abc import abstractmethod
from typing import Callable, Iterable, List, Tuple

data_dir = os.path.join('.', 'data')
haxby_dataset = datasets.fetch_haxby(subjects=[1,2,3,4,5,6], fetch_stimuli=True, data_dir=data_dir)

import numpy as np
import torch
import torch_geometric as tg


# Metrics:
from sklearn.metrics import classification_report

# Train-Test Splitter:
from sklearn.model_selection import train_test_split

# For Classical ML algorithms:
from lazypredict.Supervised import LazyClassifier





def _make_undirected(mat):
    """
    Takes an input adjacency matrix and makes it undirected (symmetric).
    Parameter
    ----------
    mat: array
        Square adjacency matrix.
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    sym = (mat + mat.transpose()) / 2
    if len(np.unique(mat)) == 2:  # if graph was unweighted, return unweighted
        return np.ceil(sym)  # otherwise return average
    return sym


def _knn_graph_quantile(mat, self_loops=False, k=8, symmetric=True):
    """
    Takes an input correlation matrix and returns a k-Nearest
    Neighbour weighted undirected adjacency matrix.
    """

    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    dim = mat.shape[0]
    if (k <= 0) or (dim <= k):
        raise ValueError("k must be in range [1,n_nodes)")
    is_directed = not (mat == mat.transpose()).all()
    if is_directed:
        raise ValueError(
            "Input adjacency matrix must be undirected (matrix symmetric)!"
        )

    # absolute correlation
    mat = np.abs(mat)
    adj = np.copy(mat)
    # get NN thresholds from quantile
    quantile_h = np.quantile(mat, (dim - k - 1) / dim, axis=0)
    mask_not_neighbours = mat < quantile_h[:, np.newaxis]
    adj[mask_not_neighbours] = 0
    if not self_loops:
        np.fill_diagonal(adj, 0)
    if symmetric:
        adj = _make_undirected(adj)
    return adj


def make_group_graph(connectomes, k=8, self_loops=False, symmetric=True):
    """
    Parameters
    ----------
    connectomes: list of array
        List of connectomes in n_roi x n_roi format, connectomes must all be the same shape.
    k: int, default=8
        Number of neighbours.
    self_loops: bool, default=False
        Wether or not to keep self loops in graph, if set to False resulting adjacency matrix
        has zero along diagonal.
    symmetric: bool, default=True
        Wether or not to return a symmetric adjacency matrix. In cases where a node is in the neighbourhood
        of another node that is not its neighbour, the connection strength between the two will be halved.
    Returns
    -------
    Torch geometric graph object of k-Nearest Neighbours graph for the group average connectome.
    """
    if connectomes[0].shape[0] != connectomes[0].shape[1]:
        raise ValueError("Connectomes must be square.")

    # Group average connectome and nndirected 8 k-NN graph
    avg_conn = np.array(connectomes).mean(axis=0)
    avg_conn = np.round(avg_conn, 6)
    avg_conn_k = _knn_graph_quantile(
        avg_conn, k=k, self_loops=self_loops, symmetric=symmetric
    )

    # Format matrix into graph for torch_geometric
    adj_sparse = tg.utils.dense_to_sparse(torch.from_numpy(avg_conn_k))
    return tg.data.Data(edge_index=adj_sparse[0], edge_attr=adj_sparse[1])




def RoI_visualizer(haxby_dataset = haxby_dataset, subject_id:int = random.randint(0,5)) -> None:
    """
        Given the subject id from i = 1,...,6, visualize the a mask of the Ventral Temporal (VT) cortex,
        coming from the Haxby with the Region of Interest (RoI) 
        
        Arguments:
        
            subject_id (int) = Subject number 
            
        Returns:
            - None  
    """
    
    # Subject ID from i = 0,...,5:
    # subject_id = 3

    # Get mask filename:
    mask_filename = haxby_dataset.mask_vt[subject_id]


    # Region of Interest Visualizations:
    plotting.plot_roi(mask_filename,
                      bg_img=haxby_dataset.anat[subject_id],
                      cmap='Paired',
                      title = f'Region of Interest of subject {subject_id}',
                      figure= plt.figure(figsize=(12,4)),
                      alpha=0.7,
                      #output_file = os.path.join(explanatory_fMRI_dir, 'roi.png'
                      )

    plotting.show()




