from Types.articleModels import FormsSet
from torch import BoolTensor
from torch.backends import mps
from torch import cuda
import numpy as np
device = "cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu"

SIGMA:dict[str, int] = {} #the IPA vocabulary
V_SIZE:int = 0
BATCH_SIZE = 5

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
        formsList (FormsSet): a matrix of forms
        add_boundaries
    Converts each string in the matrix to a tensor of one-hot tensors.
    Tensor's shape : (N, B, 1+|C|//B, |Σ|+2) if add_boundaries = False, \
    (2+N, B, 1+|C|//B, |Σ|+2) else, with
        N = max word length in the list of forms
        B = the batch size
        |C| = the size of the cognates pairs list
        |Σ| = the number of IPA characters in the vocabulary (excluding \
            special characters for the algorithm)
    """
    #TODO: converting BoolTensor to 0 1 Tensor
    max_word_length = 0
    for form in formsList:
        length = len(form)
        if length > max_word_length:
            max_word_length = length
    t = BoolTensor(size=(max_word_length+int(add_boundaries)*2,
                         BATCH_SIZE, 
                         1+len(formsList)//BATCH_SIZE, #TODO: checks if for empty strings the neural networks doesn't make mistakes.
                         V_SIZE+2), device=device)
    for n in range(len(formsList)):
        form = formsList[n]
        i, j = n%BATCH_SIZE, n//BATCH_SIZE
        if add_boundaries:
            t[0, i, j, INPUT_VOCABULARY["("]] = True
            t[len(form)+1, i, j, INPUT_VOCABULARY[")"]] = True
        for k in range(int(add_boundaries), len(form)+int(add_boundaries)):
            t[k, i, j, INPUT_VOCABULARY[form[k]]] = True
    return t
