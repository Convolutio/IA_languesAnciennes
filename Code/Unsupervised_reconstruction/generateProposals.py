import numpy as np
from Models.articleModels import Form
# calculating the minimum edit distance between the current reconstruction
# and each of its associated cognates, we add all the strings on its minimum
# edit paths to a list, which will constitute the list of proposals for the sampling

# cf. 2.5 chapter https://web.stanford.edu/~jurafsky/slp3/2.pdf for the variables notations

OPERATIONS = ("insertion", "deletion", "substitution") # we index operations likes that 

def computeMinEditDistanceMatrix(x:Form, y:Form):
    """
    We consider that a substitution has the same cost than a deletion and an insertion.
    """
    n, m = len(x), len(y)
    D = np.full((n, m), -1) # -1 = +infinity
    D[:, 0] = np.arange(n) + (1 if x[0]!=y[0] else 0)
    D[0, 1:] = np.arange(1, m) + (1 if x[0] != y[0] else 0)
    for i in range(1, n):
        for j in range(1, m):
            if j>i:
                D[i, j] = D[i,j-1]+1
            elif i>j:
                D[i, j] = D[i-1, j]+1
            else:
                editCosts = (D[i-1,j]+1, D[i, j-1]+1, 
                        D[i-1, j-1] + (0 if x[i]==y[j] else 1))
                D[i, j] = min(editCosts)
    return D

def getMinEditPaths(minEditDistanceMatrix):
    pass

print(computeMinEditDistanceMatrix(list("absɛns"), list("assɛnte")))