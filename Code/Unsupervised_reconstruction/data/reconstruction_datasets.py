from typing import Generator, Any
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset, DataLoader, TensorDataset, ChainDataset, get_worker_info
from data.vocab import computeInferenceData_Samples, computeInferenceData_Cognates, wordsToOneHots, vocabulary
from Source.utils import pad2d_sequence
from models.types import ModernLanguages, Operations, InferenceData_Samples, InferenceData_Cognates, PADDING_TOKEN

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
Raw_minibatch_type = tuple[tuple[
                        tuple[Tensor], # tuple of c ByteTensors of shape (b, |x|)
                        list[dict[ModernLanguages, Tensor]] # list of c dicts with the c cognates group
                        ],
                        tuple[int, int] # the coords of the minibatch in the global batch
                        ]

class __SamplingDataset(IterableDataset[Raw_minibatch_type]):
    """
    Returns an iterable of tuple([list of c tuple(b samples tensor, associated cognates)], coords in global batch)
    """
    def __init__(self, proposals: list[Tensor], cognates: list[dict[ModernLanguages, Tensor]],
                 samples_number_per_batch: int, cognates_batch_index:int) -> None:
        assert(len(proposals)==len(cognates)), "the two lists must have the same length to match together"
        self.proposals_dataset = TensorDataset(*proposals)
        self.cognates = cognates
        self.batch_size = samples_number_per_batch
        self.cognates_batch_index = cognates_batch_index
    
    def __iter__(self) -> Generator[Raw_minibatch_type, Any, None]:
        worker_info = get_worker_info()
        worker_id, num_workers = 0, 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        samples_batch_index = worker_id
        for start_index in range(worker_id*self.batch_size, len(self.proposals_dataset), num_workers*self.batch_size):
            yield((self.proposals_dataset[start_index:start_index+self.batch_size], self.cognates), (self.cognates_batch_index, samples_batch_index))
            samples_batch_index += num_workers

def __sampling__collate_fn(batch: Raw_minibatch_type)\
                                            -> tuple[tuple[InferenceData_Samples, dict[ModernLanguages, InferenceData_Cognates]],
                                                    tuple[int, int]]:
    collated_samples = computeInferenceData_Samples(pad_sequence(sequences=[t.T for t in batch[0][0]], padding_value=vocabulary[PADDING_TOKEN]))
    collated_cognates: dict[ModernLanguages, Tensor] = {}
    c = len(batch[0][0])
    for language in batch[0][1][0]:
        collated_cognates[language] = pad_sequence([batch[0][1][n][language] for n in range(c)], padding_value=vocabulary[PADDING_TOKEN])
    return ((collated_samples, computeInferenceData_Cognates(collated_cognates)), batch[1])

def generateSamplingDataset(proposals:list[Tensor], cognates:list[dict[ModernLanguages, Tensor]],
                            batch_shape: tuple[int, int]):
    """
    Arguments:
     - batch_shape: (c, b)
    """
    c, b = batch_shape
    datasets: list[__SamplingDataset] = []
    for i in range(len(proposals)//c):
        datasets.append(__SamplingDataset(proposals[c*i:c*(i+1)], cognates[c*i:c*(i+1)], b, i))
    return ChainDataset(datasets)

def samplingDataLoader(proposals: list[Tensor], cognates: list[dict[ModernLanguages, Tensor]], batch_shape: tuple[int, int], num_workers: int = 0):
    return DataLoader(dataset=generateSamplingDataset(proposals, cognates, batch_shape), batch_size=None, collate_fn=__sampling__collate_fn, num_workers = num_workers)
