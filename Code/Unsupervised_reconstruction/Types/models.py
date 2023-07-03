from typing import Literal
from numpy.typing import NDArray
from numpy import uint8
from torch import Tensor


EditNDArray = Tensor
EditsCombination = Tensor
Edit = tuple[Literal[0,1,2], int, int]
"""
[op, i, j]

substitution : [0, i, j]\\
deletion : [1, i, j]\\
insertion : [2, i, j]

$j$ is the index of the character of y. It enables to differentiate an operation which
takes place at different positions in the edit distance matrix, but in which i and i' have
the same value and so do y[j] and y[j'], and the operation is identical (it is a case which
has already been met). It also enables to insert characters in the same i position of x
in the good order, whatever the order with which the insertions have been applied.  
The positionnal index $i$ is between -1 and |x|-1 included, $i=-1$ meaning an insertion on
the left of the word.
"""

InferenceData = tuple[Tensor, Tensor, int]
"""
The data in a language for inferring into an edit model with the dynamic program.

This tuple contains :
    - the tensor with the batch's words with their boundaries, encoded into one-hot indexes sequence.
      (dim = (N+2, B), with B the batch size, N the variable raw sequences lengths).
      See data.vocab.py for the meaning of each one-hot indexes. 
    - the CPU IntTensor with the lengths of the raw sequences (i.e. without their boundaries). The numpy format enables CPU optimisation for the indexing in the dynamic program.
    - The max length of a raw sequence in the batch, for avoiding computing again this information for ndarray creations.
"""