import typing
from bidict import bidict

from Types.articleModels import FormsSet

import torch
from torch import Tensor
from torch.backends import mps
from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"

SIGMA: bidict[str, int] = bidict()  # the IPA vocabulary
V_SIZE: int = 0

i = 0
with open('./data/IPA_vocabulary.txt', 'r', encoding='utf-8') as vocFile:
    chars = vocFile.read().split(", ")
    V_SIZE = len(chars)
    for char in chars:
        SIGMA[char] = i
        i += 1
SIGMA_SUB, SIGMA_INS = SIGMA.copy(), SIGMA.copy()
SIGMA_SUB["<del>"], SIGMA_INS["<end>"] = i, i
INPUT_VOCABULARY = SIGMA.copy()
INPUT_VOCABULARY["("], INPUT_VOCABULARY[")"] = i, i+1


def wordToOneHots(word: str) -> torch.ByteTensor:
    """
    0: empty character\n
    1 <= i <= |Σ|: IPA character index\n
    |Σ|+k: additional special character index
    """
    return torch.ByteTensor([SIGMA[c]+1 for c in word], device=device)


def oneHotsToWord(vecSeq: torch.Tensor) -> str:
    w = ""
    for vocIdx in vecSeq:
        if vocIdx != 0:
            w += SIGMA.inverse[vocIdx.item()-1]  # type: ignore
    return w


def make_oneHotTensor(formsVec: torch.Tensor, add_boundaries: bool) -> Tensor:
    """
    Converts each string in the list to a tensor of one-hot vectors.
    Tensor's shape : (N, B, |Σ|+2) if add_boundaries = False, \
    (N+2, B, |Σ|+2) else, with
        |Σ| = the number of IPA characters in the vocabulary (excluding \
            special characters for the algorithm)

    Arguments:
        formsVec (NDArray[uint8]): a batch of forms with their one-hot indexes\
        instead of their characters. Dimension (B, N), with \
            N the maximal sequence length in the batch, \
            B the batch size
        add_boundaries (bool): if we add "(" and ")" to the forms to\
        be converted (')' is confunded with the empty character).
    """
    voc_size = V_SIZE
    batch_size, max_length = formsVec.shape
    # empty_char_index = 0
    left_boundary_index = voc_size + 1
    right_boundary_index = 0

    if not add_boundaries:
        flat = formsVec.flatten().to(torch.int64)
        tensor = torch.nn.functional.one_hot(flat, voc_size+2)
        tensor = tensor.reshape(
            (batch_size, max_length, voc_size+2)).transpose(0, 1)
    else:
        t = torch.zeros((batch_size, max_length+2), dtype=torch.uint8)
        t[:, 1:max_length+1] = formsVec
        t[:, 0] = torch.full([batch_size], voc_size+1)
        flat = t.flatten().to(torch.int64)
        tensor = torch.nn.functional.one_hot(flat, voc_size+2)
        tensor = tensor.reshape(
            (batch_size, max_length+2, voc_size+2)).transpose(0, 1)

    return tensor

def reduceOneHotTensor(t:Tensor):
    """
    Lets t a tensor with one-hot indexes and zeros on the right. This function suppress as many zeros columns as possible.
    """
    stop, i = False, t.shape[1]

    while i >= 0 and not stop:
        i -= 1
        if not torch.all(t[:, i] == 0).item():
            stop = True
    
    return t[:, :i+1]

def getWordsLengthFromOneHot(t:Tensor)->Tensor:
    """
    Returns a tensor of (batch_size) dimension with the length of each one-hot indexes-represented words in the tensor.

    Arguments:
        t (ByteTensor): a matrix representing a word at each rows. Empty characters could be present at the right. 
    """
    batch_size = t.shape[0]
    lengths = t.shape[1]*torch.ones(batch_size, dtype=torch.uint8).to(device)
    for j in range(t.shape[1]-1, -1, -1):
        jVec = j*torch.ones(batch_size, dtype=torch.uint8).to(device)
        lengths = torch.where(t[:, j]==0, jVec, lengths)
    return lengths

#TODO convertir les cognats en vecteurs one-hot une seule fois