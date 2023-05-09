from data.vocab import make_tensor
import numpy as np

def testOneHot():
    proposals = np.array([[16,  6,  6, 27, 32,  6,  0,  0,  0],
                          [16,  6,  6, 27, 31,  6,  7,  0,  0]], dtype=np.uint8)
    print(proposals.shape)
    t1 = make_tensor(proposals, False)
    t2 = make_tensor(proposals, True)
    print(t1.shape, t2.shape) # OK !