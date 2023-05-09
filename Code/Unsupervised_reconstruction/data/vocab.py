from Types.articleModels import FormsSet
from torch import Tensor
import torch
from torch.backends import mps
from torch import cuda
import numpy as np
import numpy.typing as npt
from bidict import bidict
device = "cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu"

SIGMA:bidict[str, int] = bidict() #the IPA vocabulary
V_SIZE:int = 0

i=0
with open('./data/IPA_vocabulary.txt', 'r', encoding='utf-8') as vocFile:
    chars = vocFile.read().split(", ")
    V_SIZE = len(chars)
    for char in chars:
        SIGMA[char] = i
        i+=1
SIGMA_SUB, SIGMA_INS = SIGMA.copy(), SIGMA.copy()
SIGMA_SUB["<del>"], SIGMA_INS["<end>"] = i, i
INPUT_VOCABULARY = SIGMA.copy()
INPUT_VOCABULARY["("], INPUT_VOCABULARY[")"] = i, i+1

def wordToOneHots(word:str)->npt.NDArray[np.uint8]:
    """
    0: empty character\n
    1 <= i <= |Σ|: IPA character index\n
    |Σ|+k: additional special character index
    """
    return np.array([SIGMA[c]+1 for c in word], dtype=np.int8)

def oneHotsToWord(vecSeq: npt.NDArray[np.uint8])->str:
    w = ""
    for vocIdx in vecSeq:
        if vocIdx != 0:
            w += SIGMA.inverse[vocIdx-1]
    return w

def make_tensor(formsVec:npt.NDArray[np.uint8], add_boundaries:bool)->Tensor:
    """
    Arguments:
        formsVec (NDArray[uint8]): a batch of forms with their one-hot indexes\
        instead of their characters. Dimension (B, N), with \
            N the maximal sequence length in the batch, \
            B the batch size
        add_boundaries (bool): if we add "(" and ")" to the forms to\
        be converted (')' is confunded with the empty character).
    Converts each string in the list to a tensor of one-hot vectors.
    Tensor's shape : (N, B, |Σ|+2) if add_boundaries = False, \
    (N+2, B, |Σ|+2) else, with
        |Σ| = the number of IPA characters in the vocabulary (excluding \
            special characters for the algorithm)
    """
    voc_size = V_SIZE
    batch_size, max_length = formsVec.shape
    # empty_char_index = 0
    left_boundary_index = voc_size + 1
    right_boundary_index = 0
    if not add_boundaries:
        flat = torch.from_numpy(formsVec).flatten().to(torch.int64)
        tensor = torch.nn.functional.one_hot(flat, voc_size+2)
        tensor = tensor.reshape((batch_size, max_length, voc_size+2)).transpose(0, 1)
    else:
        t = np.zeros((batch_size, max_length+2), dtype=np.uint8)
        t[:, 1:max_length+1] = formsVec
        t[:, 0] = np.full(batch_size, voc_size+1)
        flat = torch.from_numpy(t).flatten().to(torch.int64)
        tensor = torch.nn.functional.one_hot(flat, voc_size+2)
        tensor = tensor.reshape((batch_size, max_length+2, voc_size+2)).transpose(0,1)
    return tensor