from typing import Generator, Any

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset, DataLoader, TensorDataset, ChainDataset, get_worker_info

from data.vocab import computeInferenceData_Samples, computeInferenceData_Cognates, wordsToOneHots, vocabulary
from Source.utils import pad2d_sequence
from models.types import ModernLanguages, Operations, InferenceData_Samples, InferenceData_Cognates, PADDING_TOKEN


TensorCognates = dict[ModernLanguages, Tensor]

# ---------------Maximisation Datasets---------------------

RawSample = type(str)
RawCognates = dict[ModernLanguages, str]
# TODO: Rework data structure : duplication dict modernlanguages
CachedTargetProbs = dict[ModernLanguages, dict[Operations, Tensor]]


def __training__collate_fn(batch: list[tuple[RawSample, RawCognates, CachedTargetProbs]]):
    """
    Collates the input and target data in the batch.
    """
    languages = tuple(batch[0][1].keys())
    operations = batch[0][2][languages[0]].keys()
    device = batch[0][2][languages[0]]["sub"].device

    firstElement = computeInferenceData_Samples(
        wordsToOneHots([t[0] for t in batch], device))

    cognates_batch: TensorCognates = {
        lang: wordsToOneHots([t[1][lang] for t in batch], device) for lang in languages}
    secondElement = computeInferenceData_Cognates(cognates_batch)

    maxSourceLength = firstElement[2] - 1
    maxCognateLength = {lang: secondElement[lang][2] for lang in languages}
    lastElement: CachedTargetProbs = {lang: {op: pad2d_sequence([t[2][lang][op] for t in batch], 0)[
        :maxSourceLength, :maxCognateLength[lang]].squeeze(3) for op in operations} for lang in languages}

    return (firstElement, secondElement, lastElement)


class __TrainingDataset(Dataset[tuple[RawSample, RawCognates, CachedTargetProbs]]):
    def __init__(self, raw_samples: list[RawSample], raw_cognates: list[RawCognates], target_probs: list[CachedTargetProbs]) -> None:
        assert(len(raw_samples) == len(raw_cognates) == len(target_probs)), "The lists to be zipped must have the same length."

        self.training_load = list(zip(raw_samples, raw_cognates, target_probs))
        self.length = len(target_probs)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[RawSample, RawCognates, CachedTargetProbs]:
        return self.training_load[index]


def trainingDataLoader(raw_samples: list[RawSample],
                       raw_cognates: list[RawCognates],
                       target_probs: list[CachedTargetProbs],
                       mini_batch_size: int, num_workers: int = 1) \
        -> DataLoader[tuple[InferenceData_Samples, dict[ModernLanguages, InferenceData_Cognates], CachedTargetProbs]]:

    training_dataset = __TrainingDataset(raw_samples,
                                         raw_cognates,
                                         target_probs)

    """
    The type was ignored because Python takes into account
    the type returned by the dataset's '__getitem__'and 
    not the post-processing type, i.e. the type of 'collate_fn'.
    """
    return DataLoader(dataset=training_dataset, batch_size=mini_batch_size,
                      collate_fn=__training__collate_fn, shuffle=True, num_workers=num_workers)  # type:ignore


# -------------Sampling Datasets---------------------

AbsoluteCoords = tuple[int, int]
RawMiniBatch = tuple[tuple[tuple[Tensor, ...], list[TensorCognates]],
                            AbsoluteCoords]
"""
- tuple of c ByteTensors of shape (b, |x|)
- list of c dicts with the c cognates group
- the coords of the minibatch in the global batch
"""


class __SamplingDataset(IterableDataset[RawMiniBatch]):
    """
    Returns an iterable of tuple([list of c tuple(b samples tensor, associated cognates)], coords in global batch)
    """

    def __init__(self, proposals: list[Tensor], cognates: list[TensorCognates],
                 samples_number_per_batch: int, cognates_batch_index: int) -> None:
        assert (len(proposals) == len(cognates)
                ), "the two lists must have the same length to match together"

        self.proposals_dataset = TensorDataset(*proposals)
        self.cognates = cognates
        self.batch_size = samples_number_per_batch
        self.cognates_batch_index = cognates_batch_index

    def __iter__(self) -> Generator[RawMiniBatch, Any, None]:
        worker_info = get_worker_info()
        worker_id, num_workers = 0, 1

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        samples_batch_index = worker_id

        for start_index in range(worker_id*self.batch_size, len(self.proposals_dataset), num_workers*self.batch_size):
            yield ((self.proposals_dataset[start_index:start_index+self.batch_size], self.cognates), (self.cognates_batch_index, samples_batch_index))
            samples_batch_index += num_workers


def __sampling__collate_fn(batch: RawMiniBatch) \
        -> tuple[tuple[InferenceData_Samples, dict[ModernLanguages, InferenceData_Cognates]], AbsoluteCoords]:

    collated_samples = computeInferenceData_Samples(pad_sequence(
        sequences=[t.T for t in batch[0][0]], padding_value=vocabulary[PADDING_TOKEN]))

    collated_cognates: TensorCognates = {}
    c = len(batch[0][0])
    for language in batch[0][1][0]:
        collated_cognates[language] = pad_sequence(
            [batch[0][1][n][language] for n in range(c)], padding_value=vocabulary[PADDING_TOKEN])

    return ((collated_samples, computeInferenceData_Cognates(collated_cognates)), batch[1])


def generateSamplingDataset(proposals: list[Tensor], cognates: list[TensorCognates], batch_shape: AbsoluteCoords):
    """
    Arguments:
     - batch_shape: (c, b)
    """
    c, b = batch_shape
    datasets: list[__SamplingDataset] = []
    for i in range(len(proposals)//c):
        datasets.append(__SamplingDataset(
            proposals[c*i:c*(i+1)], cognates[c*i:c*(i+1)], b, i))

    return ChainDataset(datasets)


def samplingDataLoader(proposals: list[Tensor], cognates: list[TensorCognates], batch_shape: AbsoluteCoords, num_workers: int = 0):
    # it is expected that this result is true for each element in the `proposals` and `cognates` list
    assert(len(proposals) == len(cognates) <= batch_shape[0] and proposals[0].shape[0] <= batch_shape[1]), "The given batch shape is incorrect"
    # TODO: Rework the data structure the 'collate_fn' expected a list of elements but instead gave a tuple where the lists are contained inside.
    return DataLoader(dataset=generateSamplingDataset(proposals, cognates, batch_shape), batch_size=None, collate_fn=__sampling__collate_fn, num_workers=num_workers)
