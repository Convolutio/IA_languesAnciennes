import numpy as np
import numpy.typing as npt
from typing import Optional
from Types.models import *
from data.vocab import wordToOneHots
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
            edit = (0, i_start-1, j_start-1, 0)
        elif deltaCoord==(0, -1):
            # insertion
            insertionIdx = -1
            if parentEdit is not None and parentEdit[0] == 2 and parentEdit[1]==i_start-1:
                insertionIdx = parentEdit[3] - 1
            edit = (2, i_start-1, j_start-1, insertionIdx)
        else:
            # deletion
            edit = (1, i_start-1, j_start-1, 0)
        if not editsTree.include(edit):
            editsTree.connect(edit, parentEdit)
            getMinEditPaths(x, y, (minEditDistanceMatrix, i, j, editsTree, edit))
        else:
            editsTree.connect(edit, parentEdit)

    return editsTree

def editProtoForm(edits:EditsCombination, x_list:Tensor, y_list:Tensor, editDistance:int)->Tensor:
    """
    Applies the sequence of edits in argument to the x proto-form
    """
    lastIdxToBeProcessed = int(edits[0,0].item())
    b = torch.zeros([x_list.shape[0]+editDistance], dtype=torch.uint8)
    if lastIdxToBeProcessed == 0:
        b[:x_list.shape[0]] = x_list
        return b.unsqueeze(dim=0)
    maxInsertionIdx = -int(torch.min(edits[1:lastIdxToBeProcessed+1,3]).item())
    a:torch.Tensor = torch.full([x_list.shape[0]+1, maxInsertionIdx+1], 0, dtype=torch.uint8)
    a[1:, 0] = x_list
    for combiIdx in range(1,lastIdxToBeProcessed+1):
        edit = edits[combiIdx]
        idxInList = int(edit[1].item())+1
        if edit[0]==2:
            # insertion
            a[idxInList, int(edit[3].item())] = y_list[int(edit[2].item())].item()
        elif edit[0]==0:
            # substitution or deletion
            a[idxInList, 0] = y_list[int(edit[2].item())].item()
        else:
            a[idxInList, 0] = 0
    a = a.flatten()
    a = a[torch.nonzero(a!=0)]
    b[:a.shape[0]] = a[:, 0]
    return b.unsqueeze(dim=0)

@torch.jit.script
def combinationsToProposals(combinations:Tensor, x_list:Tensor, y_list:Tensor, editDistance:int):
    futures:list[torch.jit.Future[Tensor]] = []
    futures = [torch.jit.fork(editProtoForm, combinations[i], x_list, y_list, editDistance) for i in range(combinations.shape[0])]
    results = [torch.jit.wait(future) for future in futures]
    return torch.unique(torch.cat(results), dim=0)

class IncorrectResultsException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def computeProposals(x:str, y:str)->Tensor:
    """
    Arguments:
        - x (str): the proto-form
        - y (str): one of its cognates\\
    Returns a list of the proposals in one-hot indexes representation (sequences
    of vocabulary indexes)
    """
    editsTree = getMinEditPaths(x,y)
    editDistance = computeMinEditDistanceMatrix(x, y)[len(x), len(y)]
    nodesCombinations = editsTree.computeEditsCombinations()
    x_list = wordToOneHots(x)
    y_list = wordToOneHots(y)
    proposalsSet = combinationsToProposals(nodesCombinations, x_list, y_list, editDistance)
    # DEBUG
    # x_check, y_check = (torch.zeros(x_list.shape[0]+editDistance, dtype=torch.uint8),
    #                     torch.zeros(x_list.shape[0]+editDistance, dtype=torch.uint8))
    # x_check[:x_list.shape[0]] = x_list
    # y_check[:y_list.shape[0]] = y_list
    # if not y_check in proposalsSet and x_check in proposalsSet:
    #     raise IncorrectResultsException(f"y or x has not been computed. The algorithm is doing wrong.\n\
    #                 Data: (/{x}/, /{y}/)")
    return proposalsSet
