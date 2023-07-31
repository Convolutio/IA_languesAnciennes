from typing import Literal
from numpy.typing import NDArray
from numpy import uint8
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

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
      (shape = (N+2, C, B) or (N+2, C)), with C the number of cognate pairs, B the number of input linked to each cognate pair (1 if it is a cognate, else it is a number of proposals linked to each cognate), N the variable raw sequences lengths).
      See data.vocab.py for the meaning of each one-hot indexes. 
    - the CPU IntTensor with the lengths of the raw sequences (i.e. without their boundaries). The numpy format enables CPU optimisation for the indexing in the dynamic program. (shape = (C, B) or (C))
    - The max length of a raw sequence in the batch, for avoiding computing again this information for ndarray creations.
"""

SourceInferenceData = tuple[PackedSequence, Tensor, int]
"""
It is expected the source input data to be passed in an EditModel with the PackingEmbedding conversion which has already been externally applied.

Tuple arguments:
    * PackedSequence of the samples' embeddings, which are sequences of tokens with the boundaries. shape = (|x|+2, C*B, input_dim)
    * CPU IntTensor with the lengths of sequences (with boundaries, so |x|+2). shape = (C, B)
    * int with the maximum one
"""
TargetInferenceData = tuple[Tensor, Tensor, int]
"""
Tuple arguments:
    * IntTensor with the modern forms with one-hot indexes format. They are sequences of tokens with IPA chararacters and only the opening boundary. shape = (|y|+1, C)
    * CPU IntTensor with the lengths of sequences (with the opening boundary, so |y|+1). shape = (C)
    * int with the maximum one
"""