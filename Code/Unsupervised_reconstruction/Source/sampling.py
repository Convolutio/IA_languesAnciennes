import numpy as np
from torch import Tensor, tensor
import torch
from torch.nn.utils.rnn import pad_sequence
from Source.reconstructionModel import ReconstructionModel
from data.vocab import computeInferenceData_Samples, vocabulary, PADDING_TOKEN
from lm.PriorLM import PriorLM

INFTY_NEG = -1e9


def computeLaw(probs: np.ndarray) -> np.ndarray:
    """
    Returns a logarithmic probabilty distribution over all the probs.

    Args:
        probs (list[float]): the list of logarithmic unnormalized probs
    """
    totalProb = INFTY_NEG

    for prob in probs:
        totalProb = np.logaddexp(totalProb, prob)

    return np.array([prob-totalProb for prob in probs], dtype=float)


def computeUnnormalizedProbs(models: ReconstructionModel, priorLM: PriorLM, proposalsSetsList: list[Tensor], selectionIndexes: Tensor) -> Tensor:
    """
    Run the dynamic inferences in the neural edit models and inferences in the prior language model to compute p(x|{y_l : l \\in L}) over the proposals batch which will be built from the sampled indexes.
    """
    batch_size = len(proposalsSetsList)
    batch = pad_sequence([proposalsSetsList[n][int(selectionIndexes[n].item())] for n in range(
        batch_size)], batch_first=False, padding_value=vocabulary[PADDING_TOKEN])
    sourceInferenceData = computeInferenceData_Samples(batch)

    probs = priorLM.inference(sourceInferenceData)
    mutationProbs = models.forward_dynProg(sourceInferenceData)
    for branch_mutationProbs in mutationProbs.values():
        probs += branch_mutationProbs

    return torch.as_tensor(probs)


def metropolisHasting(proposalsSetsList: list[Tensor], models: ReconstructionModel, priorLM: PriorLM, iteration: int = 10**4) -> Tensor:
    """
    Sample proposals randomly from a probability distribution which will be computed progressively from forward dynamic program (so the language model, the edit models and the cognates are required).

    Args:
        proposalsSetsInfos (list[ByteTensor]) : a list of the proposals sets for each reconstruction
        models (dict[Languages, EditModel]): the dictionnary containing the ReconstructionModel for each modern language.
        priorLM (PriorLM): a language model of the proto-language
        iteration (int) : the number of proposal sampling iterations.
    """
    batch_size = len(proposalsSetsList)
    proposalsNumbers = tensor(
        [len(proposalsSet) for proposalsSet in proposalsSetsList], dtype=torch.float)

    # Computes once the context of targets in the edits models
    models.update_modernForm_context()

    print("-"*60)

    i = torch.zeros(batch_size, dtype=torch.int32)
    iProbs = computeUnnormalizedProbs(models, priorLM, proposalsSetsList, i)

    for it in range(iteration):
        # random j index for each sample
        j = torch.floor(torch.rand(batch_size) * (proposalsNumbers-1)).to(dtype=torch.int32)
        j = torch.where(j >= i, j+1, j)

        jProbs = computeUnnormalizedProbs(
            models, priorLM, proposalsSetsList, j)

        acceptation = jProbs - iProbs
        u = torch.log(torch.rand(batch_size))
        i = torch.where(u <= acceptation, j, i)
        iProbs = torch.where(u <= acceptation, jProbs, iProbs)

        if it % (iteration//100) == 0:
            print(f'Sampling: {it//(iteration//100)}%'+' '*10, end='\r')

    print('\n'+'-'*60+'\n')

    return pad_sequence([proposalsSetsList[n][i[n]] for n in range(batch_size)], batch_first=False, padding_value=vocabulary[PADDING_TOKEN])
