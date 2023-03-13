import pickle
import numpy as np

def save_obj(obj:object, path:str = None) -> None:
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        
def load_obj(path:str = None) -> object:
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)


def save(data:np.ndarray = None,path:str = None) -> None:
    np.save(path + '.npy', data, allow_pickle=True)


def load(path:str = None) -> np.ndarray:
    return np.load(path + '.npy', allow_pickle=True)   