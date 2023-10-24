from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from data.vocab import computeInferenceData_Samples, computeInferenceData_Cognates, wordsToOneHots
from source.utils import pad2d_sequence
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

class __TrainingDataset(Dataset):
    def __init__(self, raw_samples: list[str],
                 raw_cognates: list[dict[ModernLanguages, str]],
                 target_probs: list[dict[ModernLanguages, dict[Operations, Tensor]]]) -> None:
        self.training_load = list(zip(raw_samples, raw_cognates, target_probs))
        self.length = len(target_probs)

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index):
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

