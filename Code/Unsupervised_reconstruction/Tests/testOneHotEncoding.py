from data.vocab import make_oneHotTensor
from torch import ByteTensor

def testOneHot():
    proposals = ByteTensor([[16,  6,  6, 27, 32,  6,  0,  0,  0],
                          [16,  6,  6, 27, 31,  6,  7,  0,  0]])
    print(proposals.shape)
    t1 = make_oneHotTensor(proposals, False)
    t2 = make_oneHotTensor(proposals, True)
    print(t1.shape, t2.shape) # OK !