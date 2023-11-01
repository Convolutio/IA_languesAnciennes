from typing import Iterator
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, DataLoader
from data.vocab import computeInferenceData_Samples, computeInferenceData_Cognates, wordsToOneHots
from Source.utils import pad2d_sequence
from models.types import ModernLanguages, Operations, InferenceData_Samples, InferenceData_Cognates

#---------------Maximisation Datasets---------------------

__CachedTargetProbs = dict[ModernLanguages, dict[Operations, Tensor]]

def __training__collate_fn(batch:list[tuple[str, dict[ModernLanguages, str], __CachedTargetProbs]]):
    """
    Collates the input and target data in the batch.
    """
    languages = tuple(batch[0][1].keys())
    operations = batch[0][2][languages[0]].keys()
    firstElement = computeInferenceData_Samples(wordsToOneHots([t[0] for t in batch]))
    cognates_batch: dict[ModernLanguages, Tensor] = {lang: wordsToOneHots([t[1][lang] for t in batch]) for lang in languages}
    secondElement = computeInferenceData_Cognates(cognates_batch)
    maxSourceLength = firstElement[2] - 1
    maxCognateLength = {lang: secondElement[lang][2] for lang in languages}
    lastElement: __CachedTargetProbs = {lang: {op:pad2d_sequence([t[2][lang][op] for t in batch], 0)[:maxSourceLength, :maxCognateLength[lang]].squeeze(3) for op in operations} for lang in languages}

    return (firstElement, secondElement, lastElement)

class __TrainingDataset(Dataset[tuple[str,
                                      dict[ModernLanguages, str],
                                      dict[ModernLanguages, dict[Operations, Tensor]]
                                      ]]):
    def __init__(self, raw_samples: list[str],
                 raw_cognates: list[dict[ModernLanguages, str]],
                 target_probs: list[dict[ModernLanguages, dict[Operations, Tensor]]]) -> None:
        assert(len(raw_samples)==len(raw_cognates)==len(target_probs)), "The lists to be zipped must have the same length."
        self.training_load = list(zip(raw_samples, raw_cognates, target_probs))
        self.length = len(target_probs)

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index: int):
        return self.training_load[index]

def trainingDataLoader(raw_samples:list[str],
                       raw_cognates:list[dict[ModernLanguages, str]],
                       target_probs:list[__CachedTargetProbs],
                       mini_batch_size:int,
                       num_workers:int=1)\
                          -> DataLoader[tuple[InferenceData_Samples, dict[ModernLanguages, InferenceData_Cognates], __CachedTargetProbs]]:
    training_dataset = __TrainingDataset(raw_samples, raw_cognates, target_probs)
    return DataLoader(dataset=training_dataset, batch_size=mini_batch_size, collate_fn=__training__collate_fn, shuffle=True, num_workers=num_workers)


#-------------Sampling Datasets---------------------
class __SamplingDataset(IterableDataset[tuple[list[tuple[Tensor, dict[ModernLanguages, Tensor]]],
                                         tuple[int, int]
                                         ]]):
    """
    Returns an iterable of tuple([list of c tuple(b samples tensor, associated cognates)], coords in global batch)
    """
    def __init__(self, cognates_group_with_proposals: list[tuple[Tensor, dict[ModernLanguages, Tensor]]],
                 batch_shape:tuple[int, int]) -> None:
        """
        Arguments:
            - cognates_group_with_proposals: a list of tuples each containing one cognates group with its associated proposals tensor (shape = (B, L~))
            - batch_shape: the shape `(c, b)` of a batch of `c` cognates group with `b` proposals for each. 
        """
        self.cognates_group_with_proposals = cognates_group_with_proposals
        self.c, self.b = batch_shape
        self.C = len(cognates_group_with_proposals)
        self.B = len(cognates_group_with_proposals[0][0]) # self.B := total proposals number
    
    def __iter__(self) -> Iterator[tuple[list[tuple[Tensor, dict[ModernLanguages, Tensor]]],
                                         tuple[int, int]
                                         ]]:
        for i in range(self.C//self.c + bool(self.C%self.c)):
            splitted_list = []
            for n in range(self.c*i, min(self.c*(i+1), self.C)):
                splitted_list.append([(t, self.cognates_group_with_proposals[n][1]) for t in self.cognates_group_with_proposals[n][0].split(self.b)])
            for j in range(self.B//self.b + bool(self.B%self.b)):
                yield ([splitted_list[n][j] for n in range(len(splitted_list))], (i, j))

def __sampling__collate_fn(batch: tuple[list[tuple[Tensor, dict[ModernLanguages, Tensor]]],
                                        tuple[int, int]])\
                                            -> tuple[tuple[InferenceData_Samples, dict[ModernLanguages, InferenceData_Cognates]],
                                                    tuple[int, int]]:
    return batch

def samplingDataLoader(cognates_group_with_proposals: list[tuple[Tensor, dict[ModernLanguages, Tensor]]]):
    return DataLoader(dataset=__SamplingDataset(cognates_group_with_proposals, (30, 50)), batch_size=None, collate_fn=__sampling__collate_fn)