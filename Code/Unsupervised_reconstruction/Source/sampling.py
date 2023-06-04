import numpy as np
from torch import Tensor, tensor
import torch
from Types.articleModels import Languages
from Source.dynamicPrograms import compute_mutation_prob

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

def computeUnnormalizedProbs(proposalsSetList:Tensor, cognates:dict[Languages, Tensor], selectionIndexes:Tensor)->Tensor:
    batch_size = len(proposalsSetList)
    batch = proposalsSetList[:, selectionIndexes, :]
    #TODO : appeler le programme dynamique
    pass
    
def metropolisHasting(proposalsSetList: Tensor, cognates:dict[Languages, Tensor], iteration: int = 10**4) -> torch.Tensor:
    """
    Sample proposals randomly from a probability distribution computed progressively from forward dynamic programs.

    Arguments:
        proposalsSetList (ByteTensor, dim: (batch_size, maxProposalsSetLength, maxStringLength)): the list of proposals for each reconstruction (with a lot of padding).
        cognates (dict[Languages, BoolTensor]) : a tensor of one-hot vectors representing the cognates of the training dataset in each language.
        iteration (int) : the number of MH proposal sampling iterations.
    """
    batch_size = len(proposalsSetList)
    i = torch.zeros(batch_size, dtype=torch.int32)
    iProbs = computeUnnormalizedProbs(proposalsSetList, cognates, i)
    proposalsNumbers = tensor([len(proposalsSet) for proposalsSet in proposalsSetList], dtype=torch.int32)
    
    for _ in range(iteration):
        # random j index for each sample
        j = torch.floor(
            torch.dot(torch.rand(batch_size), proposalsNumbers-1)
            ).to(dtype=torch.int32)
        j = torch.where(j>=i, j+1, j)
        
        jProbs = computeUnnormalizedProbs(proposalsSetList, cognates, j)
        
        acceptation = jProbs - iProbs
        u = torch.log(torch.rand(batch_size))
        i = torch.where(u<=acceptation, j, i)
        iProbs = torch.where(u<=acceptation, jProbs, iProbs)
    #TODO
    return proposalsSetList[:, i, :]
