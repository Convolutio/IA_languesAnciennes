import time
from time import time
from torch import tensor

def functionToRun():
    """
    Call here the function
    """
    pass
    
if __name__=="__main__":
    start_time = time()
    functionToRun()
    duration = time() - start_time #in seconds

    print(f'\n\nExecution time : {duration//60} minutes and {duration%60} seconds')