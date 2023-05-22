from torch import Tensor
from typing import Literal

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