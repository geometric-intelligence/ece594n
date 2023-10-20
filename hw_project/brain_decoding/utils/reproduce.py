import numpy as np
from typing import Callable, Iterable, List   
    
    

    
def random_seed(Func:Callable):
    """
        
        Decorator random seeder.
    
    """
    def _random_seed(*args, **kwargs):
        np.random.seed(42)
        random.seed(42)
        result = Func(*args, **kwargs)
        return result
    return _random_seed
