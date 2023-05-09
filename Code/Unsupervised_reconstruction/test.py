from Tests import timeToSample
#from Tests import metropolisHastings
from Tests.testEditsGraph import variousDataTest, bigDataTest
from Tests.testOneHotEncoding import testOneHot
from numpy import array, uint8, unique

if __name__=="__main__":
    bigDataTest()