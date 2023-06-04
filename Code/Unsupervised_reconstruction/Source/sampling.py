import numpy as np
from torch import Tensor, tensor
import torch
from Types.articleModels import Languages

INFTY_NEG = -1e9


def computeLaw(probs: np.ndarray) -> np.ndarray:
    """
    Returns a logarithmic probabilty distribution over all the probs.

    Arguments:
        probs (list[float]): the list of logarithmic unnormalized probs
    """
    totalProb = INFTY_NEG

    for prob in probs:
        totalProb = np.logaddexp(totalProb, prob)

    return np.array([prob-totalProb for prob in probs], dtype=float)

def computeUnnormalizedProbs(proposalsSetList:list[Tensor], cognates:dict[Languages, Tensor], selectionIndexes:Tensor)->Tensor:
    batch_size = len(proposalsSetList)
    batch = [proposalsSetList[n][selectionIndexes[n]] for n in range(batch_size)]
    for n in range(batch_size):
        proposal = batch[n]
    #TODO
    pass
    
def metropolisHasting(proposalsSetList: list[Tensor], cognates:dict[Languages, Tensor], iteration: int = 10**4) -> int:
    """
    Sample the index of a proposal randomly from the probability
    distribution.

    Arguments:
        law (list[float]) : a probability distribution in logarithm
    """
    batch_size = len(proposalsSetList)
    i = torch.zeros(batch_size, dtype=torch.int32)
    iProbs = computeUnnormalizedProbs(proposalsSetList, cognates, i)
    lengths = tensor([len(proposalsSet) for proposalsSet in proposalsSetList], dtype=torch.int32)
    
    for _ in range(iteration):
        # random j index for each sample
        j = torch.floor(
            torch.dot(torch.rand(batch_size), lengths-1)
            ).to(dtype=torch.int32)
        j = torch.where(j>=i, j+1, j)
        
        jProbs = computeUnnormalizedProbs(proposalsSetList, cognates, j)
        
        acceptation = iProbs - jProbs
        u = torch.log(torch.rand(batch_size))
        i = torch.where(u<=acceptation, j, i)
        iProbs = torch.where(u<=acceptation, jProbs, iProbs)
    #TODO
    return []
