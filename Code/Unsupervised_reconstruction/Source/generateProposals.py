import numpy as np
import numpy.typing as npt
from typing import Optional
from Types.models import Edit
from Source.editsGraph import EditsGraph
import gc
# calculating the minimum edit distance between the current reconstruction
# and each of its associated cognates, we add all the strings on its minimum
# edit paths to a list, which will constitute the list of proposals for the sampling

# cf. 2.5 chapter https://web.stanford.edu/~jurafsky/slp3/2.pdf for the variables notations

OPERATIONS = ("insertion", "deletion", "substitution") # we index operations likes that 

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

EditsTree = EditsGraph
def getMinEditPaths(x:str, y:str,
                    recursivityArgs: Optional[tuple[npt.NDArray[np.int_], int, int, EditsTree,
                                                    Optional[Edit]]]=None) -> EditsTree:
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
        editsTree = EditsGraph()
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
        edit = None
        if deltaCoord==(-1, -1):
            # substitution
            edit = (x[i_start-1], y[j_start-1], i_start-1, 0, j_start-1)
        elif deltaCoord==(0, -1):
            # insertion
            insertionIdx = -1
            if parentEdit is not None and parentEdit[0] == "" and parentEdit[2]==i_start-1:
                insertionIdx = parentEdit[3] - 1
            edit = ("", y[j_start-1], i_start-1, insertionIdx, j_start-1)
        else:
            # deletion
            edit = (x[i_start-1], "", i_start-1, 0, j_start-1)
        if not editsTree.include(edit):
            editsTree.connect(edit, parentEdit)
            getMinEditPaths(x, y, (minEditDistanceMatrix, i, j, editsTree, edit))
        else:
            editsTree.connect(edit, parentEdit)

    return editsTree

def editProtoForm(x:str, edits:list[Edit])->str:
    """
    Applies the sequence of edits in argument to the x proto-form
    """
    a:list[list[str]] = [[]] + [list(c) for c in x]
    for edit in edits:
        idxInList = edit[2]+1
        if edit[0]=="":
            # insertion
            if len(a[idxInList]) < 1-edit[3]:
                a[idxInList] = a[idxInList][:1]+ ["" for _ in range(1-edit[3]-len(a[idxInList]))] + a[idxInList][1:] 
            a[idxInList][edit[3]] = edit[1]
        else:
            # substitution or deletion
            a[idxInList][0] = edit[1]
    output = ""
    for elt in a:
        stringified_elt = ""
        for c in elt:
            stringified_elt += c
        output += stringified_elt
    return output

def computeProposals(x:str, y:str)->list[str]:
    editsTree = getMinEditPaths(x,y)
    editDistance = computeMinEditDistanceMatrix(x, y)[len(x), len(y)]
    #TODO: restructuring the graph to minimize the repetition of computed combinations/proposals
    proposalsSet = set[str]()
    nodesCombinations = editsTree.computeNodesCombinations()
    numberOfProposals = len(nodesCombinations)
    printPercentages = numberOfProposals>100
    percent = numberOfProposals//100
    for n in range(numberOfProposals):
        nodesCombination = nodesCombinations.pop()
        editsCombination = [editsTree.getEdit(editId) for editId in nodesCombination]
        newProposal = editProtoForm(x, editsCombination)
        if len(editsCombination)==editDistance and newProposal!=y:
            print(x,y)
            print(computeMinEditDistanceMatrix(x,y))
            editsTree.displayGraph("errorGraph", "")
            print(editsCombination)
            raise Exception(f"Error: /{newProposal}/ instead of /{y}/")
        proposalsSet.add(newProposal)
        if printPercentages and n%percent==0:
            print(f'{n//percent}%')
    del(nodesCombinations)
    gc.collect()
    try:
        assert(y in proposalsSet)
    except:
        raise Exception(f"y or x has not been computed. The algorithm is doing wrong.\n\
                        Data: (/{x}/, /{y}/)")
    return list(proposalsSet)
