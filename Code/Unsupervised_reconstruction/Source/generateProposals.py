import numpy as np
import numpy.typing as npt
from typing import Optional
from Types.models import *
from data.vocab import wordToOneHots, oneHotsToWord
from Source.editsGraph import EditsGraph
import torch
from torch import Tensor

# calculating the minimum edit distance between the current reconstruction
# and each of its associated cognates, we add all the strings on its minimum
# edit paths to a list, which will constitute the list of proposals for the sampling

# cf. 2.5 chapter https://web.stanford.edu/~jurafsky/slp3/2.pdf for the variables notations

def computeMinEditDistanceMatrix(x:str, y:str)->npt.NDArray[np.int_]:
    """
    We consider that a substitution has the same cost than a deletion and an insertion.
    """
    n, m = len(x), len(y)
    D = np.full((n+1, m+1), -1) # -1 = +infinity
    D[0, :] = np.arange(m+1)
    D[1:, 0] = np.arange(1, n+1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            editCosts = (D[i-1,j]+1, D[i, j-1]+1, 
                        D[i-1, j-1] + (0 if x[i-1]==y[j-1] else 1))
            D[i, j] = min(editCosts)
    return D


def getMinEditPaths(x:str, y:str,
                    recursivityArgs: Optional[tuple[npt.NDArray[np.int_], int, int, EditsGraph,
                                                    Optional[Edit]]]=None) -> EditsGraph:
    """
    Arguments:
        x (str): the first string to be compared
        y (str): the second one
        args (tuple[IntMatrix, int, int] | None) : (minEditDistanceMatrix, i_start, j_start, editsTree, parentEdit)
    If mentionned, this recursive function figures out the minimal edit paths between x[:i_start]
    and y[:j_start] thanks to the minEditDistanceMatrix. Else, this is the minimal edit paths\
    between x and y.

    This is all the minimal edit paths with distinct editions set. A path is modeled by a recursive
    list of edits (type Edit), which modelize an arbor.
    """
    (minEditDistanceMatrix, i_start, j_start, editsTree, parentEdit) = None, -1, -1, None, None
    if recursivityArgs is not None:
        minEditDistanceMatrix, i_start, j_start, editsTree, parentEdit = recursivityArgs
    else:
        minEditDistanceMatrix = computeMinEditDistanceMatrix(x, y)
        i_start = minEditDistanceMatrix.shape[0]-1
        j_start = minEditDistanceMatrix.shape[1]-1
        editsTree = EditsGraph(x, y, minEditDistanceMatrix[len(x), len(y)])
    currentPathLength = minEditDistanceMatrix[i_start, j_start]
    if currentPathLength == 0:
        return editsTree
    possiblePriorCoords:list[tuple[int, int]] = []
    if i_start > 0:
        possiblePriorCoords.append((i_start-1, j_start))
    if j_start > 0:
        possiblePriorCoords.append((i_start, j_start-1))
        if i_start>0:
            possiblePriorCoords.append((i_start-1, j_start-1))
    minPriorPathLength = min(int(minEditDistanceMatrix[c[0], c[1]]) for c in possiblePriorCoords)
    possiblePriorCoords = [c for c in possiblePriorCoords 
                           if int(minEditDistanceMatrix[c[0], c[1]])==minPriorPathLength]
    if currentPathLength==minPriorPathLength:
        # reclimb the matrix with a neutral substitution
        return getMinEditPaths(x, y, (minEditDistanceMatrix, i_start-1, j_start-1, editsTree, parentEdit))
    
    for c in possiblePriorCoords:
        i, j = c
        deltaCoord = (i-i_start, j-j_start)
        edit:Edit
        if deltaCoord==(-1, -1):
            # substitution
            edit = (0, i_start-1, j_start-1)
        elif deltaCoord==(0, -1):
            # insertion
            edit = (2, i_start-1, j_start-1)
        else:
            # deletion
            edit = (1, i_start-1, j_start-1)
        if not editsTree.include(edit):
            editsTree.connect(edit, parentEdit)
            getMinEditPaths(x, y, (minEditDistanceMatrix, i, j, editsTree, edit))
        else:
            editsTree.connect(edit, parentEdit)

    return editsTree
class IncorrectResultsException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def removeZeros(t:Tensor)->Tensor:
    """
    Arguments:
        t (Tensor): a tensor of shape (batch_size, N)
    Rewrite each tensor's row for the zeros to all be on the right.
    """
    N = t.shape[1]
    for j in range(N-1):
        stop = False
        while not stop and not torch.all(t[:, j] != 0).item():
            a = torch.where((t[:, j]==0).repeat(t.shape[1]-j, 1).T,
                               (t[:, j:]).roll(-1, 1),
                               t[:,j:])
            if torch.all(a==t[:,j:]).item():
                stop = True
            else:
                t[:, j:] = a
    stop, i = False, N
    while i>=0 and not stop:
        i-=1
        if not torch.all(t[:,i]==0).item():
            #DEBUG
            # for j in range(i+1, nodesCombinations.shape[1]):
            #     if not torch.all(nodesCombinations[:,j]==0).item():
            #         raise IncorrectResultsException()
            stop = True
    return t[:, :i+1]

def computeProposals(x:str, y:str)->Tensor:
    """
    Arguments:
        - x (str): the proto-form
        - y (str): one of its cognates\\
    Returns a list of the proposals in one-hot indexes representation (sequences
    of vocabulary indexes)
    """
    editsTree = getMinEditPaths(x,y)
    proposalsSet = editsTree.computeEditsCombinations()
    proposalsSet = removeZeros(proposalsSet)
    proposalsSet = proposalsSet.unique(sorted=False, dim=0)
    # DEBUG
    # proposalsNumber, maxProposalLength = proposalsSet.shape
    # x_list, y_list = wordToOneHots(x), wordToOneHots(y)
    # x_check, y_check = (torch.zeros(maxProposalLength, dtype=torch.uint8),
    #                      torch.zeros(maxProposalLength, dtype=torch.uint8))
    # x_check[:x_list.shape[0]] = x_list
    # y_check[:y_list.shape[0]] = y_list
    # if not y_check in proposalsSet and x_check in proposalsSet:
    #     raise IncorrectResultsException(f"y or x has not been computed. The algorithm is doing wrong.\n\
    #                 Data: (/{x}/, /{y}/)")
    # uniques, counts = proposalsSet.unique(sorted=False, dim=0, return_counts=True)
    # if uniques.shape[0] < proposalsNumber:
    #     for i in range(uniques.shape[0]):
    #         word, count = oneHotsToWord(uniques[i]), counts[i]
    #         if count > 1:
    #             print(f"{word} ({[u.item() for u in uniques[i]]}): {count}")
    #     raise IncorrectResultsException(f"The returned proposals set has duplicated strings.\n\
    #                  Data: (/{x}/, /{y}/)")
    return proposalsSet
