from __future__ import print_function, division



# For neuroimaging:
from nibabel.testing import data_path
from nilearn import plotting as nplt
from nilearn.input_data import NiftiMasker
from nilearn import datasets
from nilearn import plotting
from nilearn.image import mean_img
from nilearn.image import index_img
import nibabel as nib
from nilearn import image


# Extras:
from abc import abstractmethod
from typing import Callable, Iterable, List, Tuple


# Basics:
import numpy as np,pandas as pd


stimuli2category = {
                        'scissors'     : 0,
                        'face'         : 1, 
                        'cat'          : 2,
                        'scrambledpix' : 3,
                        'bottle'       : 4,
                        'chair'        : 5,
                        'shoe'         : 6,
                        'house'        : 7
}

def fetch_haxby_per_subject(haxby_dataset,
                            subject_id:int = None,
                            stimulus:list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    
        Given the subject id, fetch the haxby data in matrix format.
        
        Arguments:
            - haxby_dataset     : Haxby dataset
            - subject_id (int)  : Subject number from [1,6]
            - stimulus   (list) : list of stimulus to be masked (default ignore rest condition)
            
        Returns:
            - data (Tuple[np.ndarray, np.ndarray, np.ndarray]) = Original 4-D data, Flattened + Masked Data, Label  
    
    """

        
    # Getting the data file name:
    spatio_temporal_data_path = haxby_dataset.func[subject_id]  
   
    # Getting labels:
    behavioral = pd.read_csv(haxby_dataset.session_target[subject_id], delimiter = ' ')
    
    # Creating conditional categories:
    conditions = behavioral['labels']
    
    # Creating masks for stimuli categories, (ignores rest conditions)
    condition_mask = conditions.isin(stimulus).tolist()
    
    # Appylying masks to labels (categorical):
    conditions = conditions[condition_mask]
    
    # Creating labels series (numerical):
    categories = np.array([stimuli2category[stimulus] for stimulus in conditions])
    
    # Masking fMRI images: (shape = (40, 64, 64, 864))
    fmri_niimgs = index_img(spatio_temporal_data_path, condition_mask)
    
    # Converting NumPy and transposing to (864, 40, 64, 64):
    numpy_fmri = fmri_niimgs.get_data().transpose(3,0,1,2)
    
    masker = NiftiMasker(mask_img=haxby_dataset.mask_vt[subject_id],
                         smoothing_fwhm=4,
                         standardize=True,
                         memory='nilearn_cache',
                         memory_level=1)

    masked = masker.fit_transform(fmri_niimgs)
    
    
    return numpy_fmri,  masked, categories


"""
for subject_id in range(1):
    
    # Getting the data file name:
    spatio_temporal_data_path = haxby_dataset.func[subject_id]  
   
    # Getting labels:
    behavioral = pd.read_csv(haxby_dataset.session_target[subject_id], delimiter = ' ')
    
    # Creating conditional categories:
    conditions = behavioral['labels']
    
    # Creating masks for stimuli categories, (ignores rest conditions)
    condition_mask = conditions.isin([*stimuli2category]).tolist()
    
    # Appylying masks to labels (categorical):
    conditions = conditions[condition_mask]
    
    # Creating labels series (numerical):
    categories = np.array([stimuli2category[stimulus] for stimulus in conditions])
    
    # Masking fMRI images: (shape = (40, 64, 64, 864))
    fmri_niimgs = index_img(spatio_temporal_data_path, condition_mask)
    
    # Converting NumPy and transposing to (864, 40, 64, 64):
    numpy_fmri = fmri_niimgs.get_data().transpose(3,0,1,2)
    
    masker = NiftiMasker(mask_img=haxby_dataset.mask_vt[subject_id],
                         smoothing_fwhm=4,
                         standardize=True,
                         memory='nilearn_cache',
                         memory_level=1)

    masked = masker.fit_transform(fmri_niimgs)
"""