from __future__ import print_function, division

# Basics:
import numpy as np,pandas as pd, matplotlib.pyplot as plt, seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Extras:
from abc import abstractmethod
from typing import Callable, Iterable, List


def confusion_matrix(labels:Iterable[list or np.ndarray],
                     preds:Iterable[list or np.ndarray]) -> pd.DataFrame:
    """
        Takes desireds/labels and softmax predictions,
        return a confusion matrix.
        
    """
    label = pd.Series(labels,name='Actual')
    pred = pd.Series(preds,name='Predicted')
    return pd.crosstab(label,pred)





def accuracy(labels,preds):
      return (np.sum(preds == labels) / labels.shape) * 100

    

def visualize_confusion_matrix(data:np.ndarray,
                               normalize:bool = True,
                               title:str = " ") -> None:
    
    if normalize:

        data /= np.sum(data)

    plt.figure(figsize=(15,15))
    sns.heatmap(data, 
                fmt='.2%',
                cmap = 'Greens')

    plt.title(title)
    plt.show()
