import numpy as np
import numpy.typing as npt

EditNDArray = npt.NDArray[np.int8]
EditsCombination = npt.NDArray[np.int8]
Edit = tuple[int, int, int, int]
"""
[op, i, j, k]

substitution : [0, i, j, 0]\\
deletion : [1, i, j, 0]\\
insertion : [2, i, j, k]

The insertion index $k$ is negative (or equals zero if this is not an insertion).
If we note $x'[i]$ the new string composed of $x[i]$ and several inserted characters at
the i position, and if we note y[j] one of the inserted characters with the operation
$[2, i, j, k]$, it means that $x'[i][-|k|] = y[j]$.\\
The positionnal index $i$ is between 0 and |x| included, $i=0$ meaning an insertion on
the left of the word.
$j$ is the index of the character of y. It also enable to differentiate an operation which
takes place at different positions in the edit distance matrix, but in which i and i' have
the same value and so do y[j] and y[j'], and the operation is identical (it is a case which
has already been met).
"""