import time

from Tests.timeToSample import generateProposalsOverDataset

# from Tests import metropolisHastings
from Tests.testEditsGraph import variousDataTest, bigDataTest, drawGraphs
from Tests.testOneHotEncoding import testOneHot
from numpy import array, uint8, unique

if __name__ == "__main__":
    t = time.time()
    p = generateProposalsOverDataset()
    print(((time.time() - t)/60))
