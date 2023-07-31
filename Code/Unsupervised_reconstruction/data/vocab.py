"""
One-hot encoding :
    "" -> 0 -> [0,0,...,0,0,0]
    w = Σ[i] -> i+1 -> [0,0,...,0,1,0,...,0]
                                  i
    "(" -> |Σ|+1 -> [0,0,...,0,1,0]
                              |Σ|
    ")" -> |Σ|+2 -> [0,0,...,0,0,1]
                               |Σ|+1
"""

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from torch import cuda

import numpy as np
from numpy.typing import NDArray

from Types.models import InferenceData

device = "cuda" if cuda.is_available() else "cpu"

SIGMA: dict[str, int] = {}  # the IPA vocabulary
SIGMA_INV: NDArray[np.str_] = np.array([''])
V_SIZE: int = 0

i = 1
with open('./data/IPA_vocabulary.txt', 'r', encoding='utf-8') as vocFile:
    chars = vocFile.read().split(", ")
    V_SIZE = len(chars)
    for char in chars:
        SIGMA[char] = i
        SIGMA_INV = np.append(arr=SIGMA_INV, values=[char], axis=0)
        i += 1
SIGMA_SUB, SIGMA_INS = SIGMA.copy(), SIGMA.copy()
SIGMA_SUB["<del>"], SIGMA_INS["<end>"] = i, i
INPUT_VOCABULARY = SIGMA.copy()
INPUT_VOCABULARY["("], INPUT_VOCABULARY[")"] = i, i+1


def wordToOneHots(word: str, inventory: dict[str, int] = SIGMA) -> torch.Tensor:
    """
    0: empty character\n
    1 <= i <= |Σ|: IPA character index\n
    |Σ|+k: additional special character index
    """
    return torch.ByteTensor([inventory[c] for c in word]).to(device=device)


def oneHotsToWord(batch: Tensor):
    """
    Arguments:
        - batch (Tensor): an IntTensor/LongTensor representing words in sequences of one hot indexes.
        dim = (B, L) 
    """
    # TODO
    pass
    # w = ""
    # for vocIdx in batch:
    #     if vocIdx != 0:
    #         w += SIGMA_INV[vocIdx.item()]  # type: ignore
    # return w


def make_oneHotTensor(formsVec: Tensor, add_boundaries: bool, formsLengths: NDArray[np.uint8]) -> Tensor:
    """
    Converts each string in the list to a tensor of one-hot vectors.
    Tensor's shape : (N, B, |Σ|+2) if add_boundaries = False, \
    (N+2, B, |Σ|+2) else, with
        N = the max sequence length in the batch
        B = the number of sequences in the batch (batch size)
        |Σ| = the number of IPA characters in the vocabulary (excluding \
            special characters for the algorithm)
    The sequences lengths being variable (not all equaling N or N+2), the function returns a PackedSequence
    to be directly passed in a RNN-type model.

    Arguments:
        formsVec (ByteTensor, dim = (batch_size, maxWordLength)): a batch of forms with their one-hot indexes\
        instead of their characters.
        add_boundaries (bool): if we add "(" and ")" to the forms to\
        be converted.
        formsLengths NDArray[np.uint8]: the lengths of each sequence in the batch

    Converts each one-hot index in the tensor to a one-hot vector (the zero index representing the empty character, the assiocated vector will be the null one; then, a j one-hot index will generate a one-hot vector with the 1 at the (j-1) position).
    """
    voc_size = V_SIZE
    batch_size, max_length = formsVec.shape
    # empty_char_index := 0
    left_boundary_index = voc_size+1
    right_boundary_index = voc_size+2

    # t = torch.where(formsVec != 0, formsVec-1, formsVec).cpu().numpy()
    t = formsVec
    # need numpy for fast indexing in 91st line
    if add_boundaries:
        t = np.concatenate((
            np.full((batch_size, 1), left_boundary_index, dtype=np.uint8),
            formsVec.cpu().numpy(),
            np.zeros((batch_size, 1), dtype=np.uint8)
        ), axis=1)
        t[range(batch_size), formsLengths+1] = np.full((batch_size,),
                                                       right_boundary_index, dtype=np.uint8)
        t = torch.as_tensor(t, dtype=torch.int32, device=device).T
    return t
    # flat = t.flatten().to(torch.int64)
    # tensor = torch.nn.functional.one_hot(
    #     flat, voc_size+2).to(torch.float32)  # for the LSTM module
    # tensor = tensor.reshape(
    #     (batch_size, max_length + (2 if add_boundaries else 0), voc_size+2)
    # ).transpose(0, 1)
    # # type:ignore
    # return pack_padded_sequence(tensor, formsLengths + (2 if add_boundaries else 0), batch_first=False, enforce_sorted=False)


def reduceOneHotTensor(t: Tensor):
    """
    Lets t a tensor with one-hot indexes and zeros on the right. This function suppress as many zeros columns as possible.
    """
    stop, i = False, t.shape[1]

    while i >= 0 and not stop:
        i -= 1
        if not torch.all(t[:, i] == 0).item():
            stop = True

    return t[:, :i+1]


def getWordsLengthFromOneHot(t: Tensor) -> Tensor:
    """
    Returns an int8 CPU Tensor (dim = (batch_size)) with the length of each one-hot indexes-represented words in the tensor.
    The ndarray choice is because we need cpu to fastly execute indexing with this kind of array (and also in with the method 'torch.nn.utils.rnn.pack_padded_sequebce')

    Arguments:
        t (ByteTensor): a matrix representing a word at each row. Empty characters could be present at the right. 
    """
    batch_size = t.shape[0]
    lengths = t.shape[1]*torch.ones(batch_size, dtype=torch.uint8).to(device)
    for j in range(t.shape[1]-1, -1, -1):
        jVec = j*torch.ones(batch_size, dtype=torch.uint8).to(device)
        lengths = torch.where(t[:, j] == 0, jVec, lengths)
    return lengths.cpu()


def oneHotsToWords(t: Tensor) -> list[str]:
    array = t.numpy()
    arr = np.zeros(t.shape[0], dtype=np.str_)
    for j in range(t.shape[1]):
        arr = np.char.add(arr, SIGMA_INV[array[:, j]])
    return list(arr)


def computeInferenceData(byteTensor: Tensor) -> InferenceData:
    """
    Computes data for the inference from a ByteTensor containing words in one-hot indexes format.
    To do that, the byteTensor is reduced, then the lengths of the sequences are computed and finally
    the one-hot vector encoding is carried out.

    Arguments:
        byteTensor (ByteTensor, dim=(ArbitrarySequenceLength, *)) : the tensor with the encoded words.  
    """
    voc_size = V_SIZE
    left_boundary_index = voc_size+1
    right_boundary_index = voc_size+2

    rawShape = byteTensor.size()
    withBoundariesTensor = torch.cat((
            torch.full((1, *rawShape[1:]), left_boundary_index, device=device, dtype=torch.int32),
            byteTensor,
            torch.zeros((1, *rawShape[1:]), device=device, dtype=torch.int32)
        ))
    t = torch.logical_xor(withBoundariesTensor[:-1], withBoundariesTensor[1:])
    withBoundariesTensor[1:] = torch.where(t, right_boundary_index, withBoundariesTensor[1:])
    lengths = torch.argmax(t.to(torch.uint8), dim=0)
    maxLength = int(torch.max(lengths).item())

    return (withBoundariesTensor[:maxLength+2], lengths.cpu(), maxLength)

# TODO convertir les cognats en vecteurs one-hot une seule fois
