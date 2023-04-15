from Models.articleModels import FormsSet
from torch import BoolTensor
from torch.backends import mps
from torch import cuda
import numpy as np
device = "cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu"

VOCABULARY:dict[str, int] = {}
V_SIZE:int = 0
i=0
with open('./data/vocabulary.txt', 'r', encoding='utf-8') as vocFile:
    chars = vocFile.read().split(", ")
    V_SIZE = len(chars)
    for char in chars:
        VOCABULARY[char] = i
        i+=1

def make_tensor(formsList:FormsSet, add_boundaries:bool):
    """
    Arguments:
        formsList (FormsSet): a matrix of forms
    Converts each string in the matrix to a tensor of one-hot tensors.
    """
    #TODO: converting BoolTensor to 0 1 Tensor
    #TODO: managing the boundaries adding 
    max_word_length = 0
    for i in range(formsList.shape[0]):
        for j in range(formsList.shape[1]):
            length = len(formsList[i, j])
            if length > max_word_length:
                max_word_length = length
    t = BoolTensor(size=(formsList.shape[0], formsList.shape[1], max_word_length, 3), device=device)
    for i in range(formsList.shape[0]):
        for j in range(formsList.shape[1]):
            form = formsList[i, j]
            for k in range(len(form)):
                t[i, j, k, VOCABULARY[form[k]]] = True
    return t
