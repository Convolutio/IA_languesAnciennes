import torch
from lm.PriorLM import NGramLM, RNNLM

def testNgramCounter():
    bigTensor = torch.tensor([[[0,0,1], [0,1,0], [0,0,1]],
                              [[0,0,1], [0,1,0], [0,0,0]]])

    littleTensor = torch.tensor([[0,0,1], [0,1,0]])

    return NGramLM.countSubtensorOccurrences(bigTensor, littleTensor) == 2

def testMulAndSum():
    VOC = 5
    dim = torch.randint(1, VOC, (1,))[0]

    t = torch.randint(0, VOC, (VOC,)*dim)
    shape = t.shape

    # Mul
    f = torch.full(shape, VOC)
    indices = torch.arange(shape[-1]).view(*([1] * (len(shape) - 1)), shape[-1])
    f_mul = f * indices

    # Sum
    n = t + f_mul
    t_sum = torch.sum(n, dim=-1)

    return t_sum

def testNgramCutter():
    s = torch.tensor([1,2])
    t = torch.tensor([[1,2,3], [2,1,4]])
    n = t.unfold(1, s.size()[0], 1)
    return n