import time
from abc import abstractmethod
from typing import Callable, Iterable, List


def timeit(Func:Callable):
    def _timeStamp(*args, **kwargs):
        since = time.time()
        result = Func(*args, **kwargs)
        time_elapsed = time.time() - since

        if time_elapsed > 60:
           print('Time Consumed : {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  
        else:        
          print('Time Consumed : ' , round((time_elapsed),4) , 's')
        return result
    return _timeStamp