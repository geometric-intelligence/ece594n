import nibabel as nib
import numpy as np

  
def fetch_from_haxby(haxby_dataset_path:str = None) -> np.ndarray:
    return nib.load(haxby_dataset_path).get_data()