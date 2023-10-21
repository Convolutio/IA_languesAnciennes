from torchdata.datapipes.iter import IterDataPipe
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from typing import TypeVar

from data.vocab import computeInferenceData, vocabulary, wordsToOneHots
from source.utils import pad2d_sequence
from models.models import TargetInferenceData, EOS_TOKEN, PADDING_TOKEN, InferenceData
from models.articleModels import ModernLanguages, Operations

def formatSources(batch:Tensor)->InferenceData:
    """
    batch: tensor (|x|, c, b)
    """
    return computeInferenceData(batch, vocabulary)

def formatTargets(batch:dict[ModernLanguages, list[str]]) -> dict[ModernLanguages, TargetInferenceData]:
    d = {}
    for lang in batch:
        targets, rawLengths, maxLength = computeInferenceData(wordsToOneHots(batch[lang], vocabulary), vocabulary)
        d[lang] = (targets.where(targets != vocabulary[EOS_TOKEN], vocabulary[PADDING_TOKEN])[:-1],
            rawLengths+1, maxLength+1)
    return d

CognateDType = TypeVar("CognateDType")
def deep_batch(batch: tuple[int, list[tuple[Tensor, dict[ModernLanguages, CognateDType]]]], b: int)\
    -> list[list[tuple[Tensor, dict[ModernLanguages, CognateDType], tuple[int, int]]]]:
    """
    Arguments:
        batch: a tuple containing:
            - the i index of the mini-batch in the axis of the cognate sets
            - a list of c tuples containing 
                - an IntTensor of shape (L, B) for B samples
                - the corresponding cognates pair (dict[ModernLanguages, CognateDType])
        b: the number of samples by cognate pair to be processed in the same mini-batch
    Returns a matrix (list of list) of shape (B//b, c) whose elements are tuples containing
        - an IntTensor of shape (L, b) for b samples
        - a dictionnary with the corresponding cognates pair
        - a tuple with the coordinates (i, j) in the main batch matrix of shape (C//c, B//b)
    """
    i, realBatch = batch
    c = len(realBatch)
    t = [elt[0].split(b, 1) for elt in realBatch] # list of c lists of n tensors with shape (L, b)
    n = len(t[0]) # = B//b + (1 if B%b != 0 else 0)
    lstToReturn = [] # list of n lists of c tuples with (tensors with shape (L, b), the cognate, (i, j))
    for j in range(n):
        lstToReturn.append([(t[a][j], realBatch[a][1], (i, j)) for a in range(c)])
    return lstToReturn

def __collate_fn(batch: list[tuple[Tensor, dict[ModernLanguages, CognateDType]]])\
    -> tuple[Tensor, dict[ModernLanguages, list[CognateDType]]]:
    """
    Arguments:
        - batch : a list of c tuples with (a tensor of shape (L,b), a cognate pair)
    Returns a tuple with (tensor of shape (L, c, b), a dict with the collated cognates pairs)
    """
    languages = batch[0][1].keys()
    return (
        pad_sequence([t[0] for t in batch], batch_first=False, padding_value=vocabulary[PADDING_TOKEN]),
        {language: [t[1][language] for t in batch] for language in languages}
    )

def __samplingCollate_fn(batch: list[tuple[Tensor, dict[ModernLanguages, CognateDType], tuple[int, int]]])\
    -> tuple[Tensor, dict[ModernLanguages, list[CognateDType]], tuple[int, int]]:
    i, j = batch[0][2]
    return (*__collate_fn([t[:2] for t in batch]), (i, j))

def __samplingBatcher(proposalsDP: IterDataPipe[Tensor],
                    cognatesDP: IterDataPipe[dict[ModernLanguages, CognateDType]],
                    batchShape: tuple[int, int]) -> IterDataPipe[tuple[
                        Tensor, dict[ModernLanguages, list[CognateDType]], tuple[int, int]
                        ]]:
    """
    Arguments:
        - proposalsDP (IterDataPipe[Tensor]): a datapipe whose elements are IntTensors of shape (L~, B) which contains the proposals for a given cognates pair.
        - cognatesDP (IterDataPipe[dict[ModernLanguages, CognateDType]]): a datapipe with the separated cognates sets corresponding to the samples
        - batchShape (tuple[int, int]): the number of cognate pairs and the number of samples by cognate pair to be processed silmutaneously.
    From a datapipe of a big batch of samples and another of their corresponding cognates pairs, returns an iterable DataPipe whose elements are mini batches of samples in the format of a tuple with the following elements:
        - an IntTensor with the samples (shape (L~, c, b))
        - a dictionnary with the collated corresponding cognates in each modern language
        - the position of the mini batch in the big batch
    """
    c, b = batchShape
    dp = proposalsDP.zip(cognatesDP).batch(c).enumerate()
    return dp.map(partial(deep_batch, b=b)).unbatch(1).collate(__samplingCollate_fn)

def samplingDataPipe(proposalsDP: IterDataPipe[Tensor],
                    cognatesDP: IterDataPipe[dict[ModernLanguages, CognateDType]],
                    batchShape: tuple[int, int]) -> IterDataPipe[tuple[
                        InferenceData, dict[ModernLanguages, list[TargetInferenceData]], tuple[int, int]
                        ]]:
    return __samplingBatcher(proposalsDP, cognatesDP, batchShape).map(formatSources, 0).map(formatTargets,1)


CachedTargetProbs = dict[ModernLanguages, dict[Operations, Tensor]]

def __training__collate_fn(batch:list[tuple[str, dict[ModernLanguages, str], CachedTargetProbs]], mini_batch_size:int):
    """
    Collates the input and target data in the batch.
    """
    languages = tuple(batch[0][1].keys())
    operations = batch[0][2][languages[0]].keys()
    firstElement = computeInferenceData(wordsToOneHots([t[0] for t in batch]), vocabulary)
    secondElement = formatTargets({lang:[t[1][lang] for t in batch] for lang in languages})
    lastElement: CachedTargetProbs = {lang: {op:pad2d_sequence([t[2][lang][op] for t in batch], 0).squeeze(3) for op in operations} for lang in languages}

    return (firstElement, secondElement, lastElement)

def get_training_datapipe(training_dp: IterDataPipe[tuple[
    str,
    dict[ModernLanguages, str],
    CachedTargetProbs
    ]], mini_batch_size:int) -> IterDataPipe[tuple[InferenceData, dict[ModernLanguages, TargetInferenceData], CachedTargetProbs]]:
        
        new_dp = training_dp.shuffle().batch(mini_batch_size).in_batch_shuffle().sharding_filter()
        return new_dp.map(partial(__training__collate_fn, mini_batch_size=mini_batch_size))

