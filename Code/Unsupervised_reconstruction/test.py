from Tests.timeToSample import generateProposalsOverDataset
# from Tests import metropolisHastings
from Tests.testEditsGraph import variousDataTest, bigDataTest, drawGraphs
from Tests.testOneHotEncoding import testOneHot
from numpy import array, uint8, unique
from time import time

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