import time
import itertools
from time import time
from Tests.ngramOperations import *

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