from Types.articleModels import FormsSet
from torch import BoolTensor
from torch.backends import mps
from torch import cuda
import numpy as np
device = "cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu"

SIGMA:dict[str, int] = {} #the IPA vocabulary
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

def make_tensor(formsList:list[str], add_boundaries:bool)->BoolTensor:
    """
    Arguments:
        formsList (list[str]): a list of forms
        add_boundaries (bool): if we add "(" and ")" to the string to\
        the string to be converted.
    Converts each string in the list to a tensor of one-hot vectors.
    Tensor's shape : (N, B, |Σ|+2) if add_boundaries = False, \
    (N+2, B, |Σ|+2) else, with
        N = max word length in the list of forms\n
        B = the batch size (=len(formsList))\n
        |C| = the size of the cognates pairs list\n
        |Σ| = the number of IPA characters in the vocabulary (excluding \
            special characters for the algorithm)
    """
    #TODO: converting BoolTensor to 0 1 Tensor
    max_word_length = 0
    batch_size = len(formsList)
    for form in formsList:
        length = len(form)
        if length > max_word_length:
            max_word_length = length
    t = BoolTensor(size=(max_word_length+int(add_boundaries)*2,
                         batch_size, 
                         V_SIZE+2), device=device)
    for n in range(batch_size):
        form = formsList[n]
        if add_boundaries:
            t[0, n, INPUT_VOCABULARY["("]] = True
            t[len(form)+1, n, INPUT_VOCABULARY[")"]] = True
        for k in range(int(add_boundaries), len(form)+int(add_boundaries)):
            t[k, n, INPUT_VOCABULARY[form[k]]] = True
    return t
