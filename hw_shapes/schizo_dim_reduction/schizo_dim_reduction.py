#Title: Visualization of the Brain Structure in Schizophrenia using Dimension Reduction Analysis
#By: Shaun Chen
import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import geomstats.visualization as visualization
import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.geometry.pre_shape import PreShapeSpace, KendallShapeMetric

visualization.tutorial_matplotlib()

data = pandas.read_csv('book-schizo.csv').to_numpy().reshape(13,28,2) #read csv file and convert to numpy array, 364x2 reshaped into 13x2x28
labels = np.append(np.zeros(14, dtype=int), np.ones(14, dtype=int)) #create labels for each class

print(data.shape)
print(labels.shape)

def plot_single():
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.scatter(data[:,0,0],data[:,0,1], marker='x') #1st normal
    plt.scatter(data[:,14,0],data[:,14,1], marker='o') #1st schizo
    plt.legend(['1st Control', '1st Schizophrenia'])
    plt.show()
    
def plot_all():
    plt.scatter(data[:,:14,0],data[:,:14,1], marker='x') #all normals
    plt.scatter(data[:,14:,0],data[:,14:,1], marker='o') #all shizos
    plt.legend(['All Control', 'All Schizophrenic'])
    plt.show()
    
def plot_corr():
    labels_str = ["Healthy", "Schizophrenic"]

    fig = plt.figure(figsize=(8, 4))

    corr_matrix = np.corrcoef(data[:,0,:])

    ax = fig.add_subplot(121)
    imgplot = ax.imshow(np.corrcoef(data[:,0,:])); ax.set_title(labels_str[labels[0]])
    ax = fig.add_subplot(122)
    imgplot = ax.imshow(np.corrcoef(data[:,14,:])); ax.set_title(labels_str[labels[14]])

    plt.show()