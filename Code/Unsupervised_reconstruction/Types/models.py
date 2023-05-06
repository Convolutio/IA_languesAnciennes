Edit = tuple[str, str, int, int, int]
"""
(a, b, i, k, j)

deletion : (a, "", i, 0, j)\\
substitution : (a, b, i, 0, j)\\
insertion : ("", b, i, k, j)

The insertion index $k$ is negative (or equals zero if this is not an insertion).
If we note $x'[i]$ the new string composed of $x[i]$ and several inserted characters at
the i position, and if we note $b$ one of the inserted characters with the operation
$("", b, i, k)$, it means that $b = x'[i][-|k|]$.\\
The positionnal index $i$ is between 0 and |x|-1 but can also equal -1, for an insertion on
the left of the word.
$j$ is the index of the character of $b$ in y, to differentiate an operation which takes place
at different positions in the edit distance matrix, but in which a, b and i are the same (it is a
case which has been met).
"""