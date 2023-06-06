import numpy as np
from torch import Tensor, tensor
import torch
from Types.articleModels import ModernLanguages
from Types.models import InferenceData
from Source.dynamicPrograms import compute_mutation_prob
from Source.editModel import EditModel
from data.vocab import computeInferenceData
from lm.PriorLM import PriorLM

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


def computeUnnormalizedProbs(models:dict[ModernLanguages, EditModel], priorLM:PriorLM, proposalsSetsList:list[Tensor], cognatesInferenceData:dict[ModernLanguages, InferenceData], selectionIndexes:Tensor, maxWordLengths:Tensor, maxWordLength:int)->Tensor:
    """
    Run the dynamic inferences in the neural edit models and inferences in the prior language model to compute p(x|{y_l : l \\in L}) over the proposals batch which will be built from the sampled indexes.
    """
    batch_size = len(proposalsSetsList)
    batch = torch.zeros((batch_size, maxWordLength), dtype=torch.uint8)
    for n in range(batch_size): batch[n, :maxWordLengths[n].item()] = proposalsSetsList[n][selectionIndexes[n]]
    sourceInferenceData = computeInferenceData(batch)

    probs = priorLM.inference(batch) #TODO: develop this method
    for language in models:
        probs += compute_mutation_prob(models[language], sourceInferenceData, cognatesInferenceData[language])
    return torch.as_tensor(probs)

    
def metropolisHasting(proposalsSetsList: list[Tensor], models:dict[ModernLanguages, EditModel], priorLM:PriorLM, cognates:dict[ModernLanguages, InferenceData], iteration: int = 10**4) -> Tensor:
    """
    Sample proposals randomly from a probability distribution which will be computed progressively from forward dynamic program (so the language model, the edit models and the cognates are required).

    Arguments:
        proposalsSetsInfos (list[ByteTensor]) : a list of the proposals sets for each reconstruction
        models (dict[Languages, EditModel]): the dictionnary containing the EditModels for each modern language.
        priorLM (PriorLM): a language model of the proto-language
        cognates (dict[Languages, BoolTensor]) : a tensor of one-hot vectors representing the cognates of the training dataset in each language.
        iteration (int) : the number of proposal sampling iterations.
    """
    batch_size = len(proposalsSetsList)
    proposalsNumbers = tensor([len(proposalsSet) for proposalsSet in proposalsSetsList], dtype=torch.int32)
    maxWordLength = 0
    maxWordLengths = torch.zeros(batch_size, dtype=torch.uint8)
    for n in range(batch_size):
        l = len(proposalsSetsList[n])
        maxWordLengths[n] = l
        if l > maxWordLength: maxWordLength=l
    
    i = torch.zeros(batch_size, dtype=torch.int32)
    iProbs = computeUnnormalizedProbs(models, priorLM, proposalsSetsList, cognates, i, maxWordLengths, maxWordLength)
    
    for _ in range(iteration):
        # random j index for each sample
        j = torch.floor(
            torch.dot(torch.rand(batch_size), proposalsNumbers-1)
            ).to(dtype=torch.int32)
        j = torch.where(j>=i, j+1, j)
        
        jProbs = computeUnnormalizedProbs(models, priorLM, proposalsSetsList, cognates, j, maxWordLengths, maxWordLength)
        
        acceptation = jProbs - iProbs
        u = torch.log(torch.rand(batch_size))
        i = torch.where(u<=acceptation, j, i)
        iProbs = torch.where(u<=acceptation, jProbs, iProbs)

    samples = torch.zeros((batch_size, maxWordLength), dtype=torch.uint8)
    for n in range(batch_size): samples[n, :maxWordLengths[n]] = proposalsSetsList[n][i[n]]
    
    return samples