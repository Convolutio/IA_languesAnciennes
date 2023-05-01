import numpy as np
import numpy.typing as npt
from typing import Union, Any
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

Edit = tuple[str, str, int]
EditsTree = list[Union[Edit, list[Any]]]
EditsBranch = list[Edit]
# del : (a, "", i)
# sub : (a, b, i)
# ins : ("", b, i)
def getMinEditPaths(x:str, y:str, 
                    args: Union[tuple[npt.NDArray[np.int_], int, int], None] = None) -> EditsTree:
    """
    Arguments:
        x (str): the first string to be compared
        y (str): the second one
        args (tuple[IntMatrix, int, int] | None) : (minEditDistanceMatrix, i_start, j_start)
    If mentionned, this recursive function figures out the minimal edit paths between x[:i_start]
    and y[:j_start] thanks to the minEditDistanceMatrix. Else, this is the minimal edit paths\
    between x and y.

    This is all the minimal edit paths with distinct editions set. A path is modeled by a recursive
    list of edits (type Edit), which modelize an arbor.
    """
    minEditDistanceMatrix, i_start, j_start = None, -1, -1
    if args is not None:
        minEditDistanceMatrix, i_start, j_start = args
    else:
        minEditDistanceMatrix = computeMinEditDistanceMatrix(x, y)
        i_start = minEditDistanceMatrix.shape[0]-1
        j_start = minEditDistanceMatrix.shape[1]-1
    currentPathLength = minEditDistanceMatrix[i_start, j_start]
    if currentPathLength == 0:
        return []
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
        return getMinEditPaths(x, y, (minEditDistanceMatrix, i_start-1, j_start-1))
    lstToReturn = []
    for c in possiblePriorCoords:
        i, j = c
        deltaCoord = (i-i_start, j-j_start)
        edit = None
        if deltaCoord==(-1, -1):
            # substitution
            edit = (x[i_start-1], y[j_start-1], i_start-1)
        elif deltaCoord==(0, -1):
            # insertion
            edit = ("", y[j_start-1], i_start-1)
        else:
            # deletion
            edit = (x[i_start-1], "", i_start-1)
        lstToReturn.append( [edit] + getMinEditPaths(x,y, (minEditDistanceMatrix, i, j)) )
    return lstToReturn if len(lstToReturn) > 1 else lstToReturn[0] #type:ignore

def editProtoForm(x:str, edits:list[Edit])->str:
    """
    Applies the sequence of edits in argument to the x proto-form
    """
    a = list(x)
    toBeInsertedToLeft = ""
    for edit in edits:
        if edit[0]=="":
            # insertion
            if edit[2]==-1:
                toBeInsertedToLeft = edit[1]
            else:
                a[edit[2]] = a[edit[2]]+edit[1]
        else:
            # substitution or deletion
            a[edit[2]] = edit[1]+a[edit[2]][1:]
    output = ""
    for elt in a:
        output += elt
    return toBeInsertedToLeft + output

def addProposalsWithDFS(x:str, editsTree:EditsTree, currentCombinationsList:list[list[Edit]], proposalsSet:set[str]):
    """
    Side effect function which add proposals to the proposal set in argument.
    It do it with a depth-first search in the edits tree in argument, initially\
    computed with from a search in the edit distance computing matrix.
    Attention!: finally the cognat will be added to the proposalSet (the combination
    of d edits will be processed.)
    """
    newCombinationsList = currentCombinationsList.copy()
    firstElt = editsTree[0]
    if isinstance(firstElt, tuple):
        proposalsSet.add(editProtoForm(x, [firstElt]))
        newCombinationsList.append([firstElt])
        for combinations in currentCombinationsList:
            newEditsCombination = combinations+[firstElt]
            # the cognate is wrongly added |B| times!!!
            # |B| is the number of different edit paths of a length of d
            # d is the minimum edit path between x and y
            proposalsSet.add(editProtoForm(x, newEditsCombination))
            newCombinationsList.append(newEditsCombination)
        if len(editsTree)>1:
            newEditsTree = editsTree[1:]
            addProposalsWithDFS(x, newEditsTree, newCombinationsList, proposalsSet)
    else:
        addProposalsWithDFS(x, firstElt, currentCombinationsList, proposalsSet)
        if isinstance(editsTree[1], tuple) :raise Exception("The tree is not built as expected")
        addProposalsWithDFS(x, editsTree[1], currentCombinationsList, proposalsSet)

def computeProposals(x:str, y:str)->list[str]:
    editsTree = getMinEditPaths(x,y)
    proposalsSet = set[str]()
    addProposalsWithDFS(x, editsTree, [], proposalsSet)
    try:
        proposalsSet.remove(y)
    except KeyError:
        raise Exception(f"({x}, {y}), {str(proposalsSet)}")
    return list(proposalsSet)
    
if __name__ == "__main__":
    x, y = "lɛɡˈatɪɔ", "leɡasjˈɔ̃"
    x1, y1 = "absɛns", "assɛnte"
    a, b = "abɛrɾasɔ", "aberɾatsiˈone"
    print(computeMinEditDistanceMatrix(a, b))
    print(getMinEditPaths(a, b))
   # print(computeProposals(a,b))

"""
# "absɛns" to "assɛnte"
a:EditsTree = [
        [('', 'e', 5, 6), ('s', 't', 5, 5), ('b', 's', 1, 1)], 
        [('s', 'e', 5, 6), ('', 't', 4, 5), ('b', 's', 1, 1)]
    ]

# ? to ? 
b:EditsTree = [
    [('s', '', 10, 8), ('n', '̃', 9, 8), ('ɛ', 'ɑ', 8, 7), ('k', 'ˈ', 7, 6), ('ˈ', '', 4, 3)],
    [('s', '̃', 10, 8), [('n', '', 9, 7), ('ɛ', 'ɑ', 8, 7), ('k', 'ˈ', 7, 6), ('ˈ', '', 4, 3)], 
                       [('n', 'ɑ', 9, 7), [('ɛ', '', 8, 6), ('k', 'ˈ', 7, 6), ('ˈ', '', 4, 3)],
                                          [('ɛ', 'ˈ', 8, 6), ('k', '', 7, 5), ('ˈ', '', 4, 3)]
                                                                                             ]
                                                                                             ]
    ]
# "lɛɡˈatɪɔ" to "leɡasjˈɔ̃"
c:EditsArbor = [('', '̃', 7, 8), [('', 'ˈ', 6, 6), ('ɪ', 'j', 6, 5), ('t', 's', 5, 4), ('ˈ', '', 3, 2), ('ɛ', 'e', 1, 1)],
                                [('ɪ', 'ˈ', 6, 6), [('',, 'j', 5, 5), ('t', 's', 5, 4), ('ˈ', '', 3, 2), ('ɛ', 'e', 1, 1)],
                                                    [('t', 'j', 5, 5), [('', 's', 4, 4), ('ˈ', '', 3, 2), ('ɛ', 'e', 1, 1)], 
                                                                        [('a', 's', 4, 4), ('ˈ', 'a', 3, 3), ('ɛ', 'e', 1, 1)]
                                                                                                                            ]
                                                                                                                            ]
                ]

"""
a = [
    [('', 'e', 7), 
        [('', 'n', 7), 
            [('', 'o', 7), 
                [('', 'ˈ', 7), ('ɔ', 'i', 7), ('', 't', 5), ('ɛ', 'e', 2)], 
                [('ɔ', 'ˈ', 7), ('', 'i', 6), ('', 't', 5), ('ɛ', 'e', 2)]
            ],
            [('ɔ', 'o', 7), ('', 'ˈ', 6), ('', 'i', 6), ('', 't', 5), ('ɛ', 'e', 2)]
        ],
        [('ɔ', 'n', 7), ('', 'o', 6), ('', 'ˈ', 6), ('', 'i', 6), ('', 't', 5), ('ɛ', 'e', 2)]],
    [('ɔ', 'e', 7), ('', 'n', 6), ('', 'o', 6), ('', 'ˈ', 6), ('', 'i', 6), ('', 't', 5), ('ɛ', 'e', 2)]]

for elt in ['abɛrɾatsnˈɔ', 'abɛrɾasɔenoˈ', 'abɛrɾatsio', 'aberɾatsioe', 'aberɾasˈioen', 'aberɾasoˈn', 'aberɾasnoˈie', 'aberɾatsˈiɔen', 'abɛrɾatsˈen', 'aberɾatsniɔ', 'abɛrɾasˈiɔen', 'aberɾatso', 'abɛrɾatsoˈe', 'aberɾasnoˈe', 'aberɾatsnˈɔ', 'aberɾasinoˈ', 'aberɾasniɔ', 'abɛrɾasine', 'aberɾasoɔe', 'abɛrɾasɔeoˈ', 'aberɾatse', 'aberɾasoiɔ', 'aberɾatsnˈe', 'aberɾasɔo', 'abɛrɾasie', 'aberɾatson', 'aberɾatsɔnˈ', 'abɛrɾatsˈine', 'aberɾatsienˈ', 'aberɾasoɔ', 'abɛrɾatsɔen', 'aberɾatsɔeno', 'abɛrɾasɔnˈ', 'abɛrɾatsɔenoˈ', 'aberɾasinˈ', 'aberɾasˈɔ', 'aberɾasnˈiɔ', 'aberɾatsnoˈe', 'abɛrɾasoˈe', 'abɛrɾatsie', 'abɛrɾatsienˈ', 'aberɾatsˈin', 'aberɾasˈoen', 'aberɾatsiɔ', 'abɛrɾatsiˈ', 'aberɾasoen', 'abɛrɾatsoˈin', 'abɛrɾatsoˈie', 'abɛrɾatsiɔen', 'aberɾatsieoˈ', 'aberɾasn', 'aberɾatsi', 'aberɾatsˈiɔ', 'aberɾasˈion', 'abɛrɾatsnie', 'abɛrɾasnˈiɔ', 'abɛrɾatsnɔ', 'aberɾatsiɔe', 'aberɾasoie', 'abɛrɾatse', 'aberɾasɔˈ', 'abɛrɾasˈiɔe', 'abɛrɾatsɔno', 'abɛrɾatsˈɔen', 'aberɾasnˈɔ', 'abɛrɾasioe', 'aberɾasienˈ', 'abɛrɾasnoˈiɔ', 'aberɾasˈɔe', 'aberɾatsoin', 'abɛrɾasioˈ', 'abɛrɾatsnoie', 'abɛrɾasien', 'abɛrɾasiˈe', 'abɛrɾasˈɔe', 'abɛrɾasˈno', 'abɛrɾatsoiɔe', 'aberɾatsˈoe', 'abɛrɾatsone', 'aberɾasoˈɔe', 'aberɾatsˈɔen', 'abɛrɾatsnˈiɔ', 'aberɾasɔeno', 'abɛrɾatsioˈ', 'aberɾasiˈo', 'aberɾatsiˈn', 'aberɾasɔeo', 'aberɾasieo', 'abɛrɾasniɔ', 'aberɾatsˈ', 'aberɾatsoˈɔ', 'abɛrɾatsoin', 'abɛrɾatsoˈiɔe', 'abɛrɾatsˈie', 'aberɾasˈen', 'abɛrɾasoˈɔe', 'abɛrɾasino', 'abɛrɾatsnˈie', 'aberɾatsoiɔ', 'abɛrɾatsiɔeo', 'abɛrɾatsiɔo', 'abɛrɾasoˈiɔe', 'aberɾasˈiɔe', 'abɛrɾasiˈo', 'abɛrɾason', 'aberɾatsien', 'abɛrɾatsnoˈɔ', 'abɛrɾasoˈɔ', 'aberɾatsˈion', 'aberɾasiˈn', 'aberɾasnie', 'aberɾasieoˈ', 'abɛrɾatsiˈeno', 'aberɾatsoˈine', 'aberɾasnoɔ', 'abɛrɾatsiˈno', 'aberɾasiɔe', 'abɛrɾatsˈioen', 'aberɾatsɔenoˈ', 'abɛrɾatsiɔe', 'aberɾatsiˈe', 'abɛrɾasoˈine', 'abɛrɾasɔnoˈ', 'aberɾasiˈe', 'abɛrɾatsˈeno', 'abɛrɾatsion', 'aberɾatsiˈen', 'abɛrɾatsnˈe', 'aberɾatsiˈ', 'abɛrɾasˈion', 'abɛrɾatson', 'abɛrɾasiˈen', 'aberɾason', 'abɛrɾasˈɔen', 'abɛrɾasˈne', 'aberɾatsɔo', 'aberɾatsˈne', 'aberɾatsieno', 'aberɾatsoˈie', 'abɛrɾatsino', 'aberɾasɔen', 'abɛrɾasinoˈ', 'aberɾatsnoˈie', 'abɛrɾatsiˈeo', 'abɛrɾatsoɔ', 'abɛrɾasˈin', 'aberɾatsɔeˈ', 'aberɾasio', 'abɛrɾatsioe', 'aberɾatsˈo', 'abɛrɾasone', 'aberɾasie', 'aberɾatsˈoen', 'aberɾasoˈɔ', 'aberɾatsnoe', 'aberɾatsˈiɔe', 'abɛrɾasoe', 'aberɾasne', 'aberɾatsˈeno', 'abɛrɾatsɔˈ', 'abɛrɾatsˈeo', 'aberɾatsn', 'aberɾasoˈine', 'abɛrɾatsɔeoˈ', 'aberɾatsine', 'aberɾasˈ', 'aberɾasone', 'abɛrɾasˈen', 'aberɾatsɔoˈ', 'aberɾasien', 'aberɾasˈoe', 'abɛrɾatsnoɔ', 'aberɾatsio', 'aberɾasioe', 'aberɾatsɔno', 'abɛrɾatsiɔn', 'aberɾatsˈon', 'abɛrɾasoˈne', 'aberɾasˈin', 'abɛrɾasoɔ', 'aberɾatsnˈie', 'abɛrɾatsieˈ', 'aberɾatsnɔ', 'abɛrɾatsoˈn', 'abɛrɾatsɔoˈ', 'abɛrɾatsˈiɔn', 'abɛrɾasɔen', 'abɛrɾasiɔn', 'abɛrɾasnie', 'aberɾatsoˈe', 'abɛrɾasiɔo', 'aberɾasɔeˈ', 'abɛrɾatsɔeo', 'abɛrɾatsnoˈiɔ', 'aberɾasiɔeo', 'aberɾatsoˈiɔ', 'aberɾatsone', 'abɛrɾatsˈɔ', 'abɛrɾasɔo', 'aberɾasoe', 'abɛrɾasnˈe', 'aberɾasiɔn', 'abɛrɾasoˈin', 'abɛrɾasnoˈɔ', 'aberɾasieno', 'aberɾatsne', 'aberɾasnoˈɔ', 'abɛrɾasnoiɔ', 'abɛrɾatsˈiɔen', 'abɛrɾasɔeˈ', 'abɛrɾasoin', 'aberɾasnoie', 'abɛrɾatsˈɔe', 'aberɾatsˈɔe', 'abɛrɾasieoˈ', 'aberɾasiɔen', 'aberɾasˈon', 'aberɾatsiɔeo', 'aberɾasion', 'aberɾatsɔeo', 'abɛrɾatsiˈo', 'abɛrɾasoˈie', 'abɛrɾatsine', 'abɛrɾatsieoˈ', 'abɛrɾasˈe', 'abɛrɾasienoˈ', 'aberɾasiˈ', 'abɛrɾasoine', 'abɛrɾaso', 'aberɾasnˈe', 'aberɾaso', 'aberɾatsoˈne', 'abɛrɾatsnoˈie', 'abɛrɾasˈie', 'aberɾasnoˈiɔ', 'aberɾatsoɔe', 'abɛrɾasiˈno', 'aberɾatsiˈeno', 'aberɾatsˈen', 'aberɾasɔn', 'abɛrɾatsnoˈe', 'aberɾasˈɔn', 'abɛrɾatsoine', 'aberɾatsɔ', 'abɛrɾatsien', 'aberɾasino', 'aberɾatsɔenˈ', 'aberɾasoine', 'abɛrɾasieˈ', 'abɛrɾasnoˈe', 'aberɾasoin', 'abɛrɾatsˈioe', 'abɛrɾasieo', 'aberɾasˈn', 'abɛrɾasoɔe', 'aberɾasnˈie', 'aberɾasienoˈ', 'abɛrɾasiɔeo', 'abɛrɾasiɔno', 'aberɾasoˈin', 'abɛrɾatsoˈiɔ', 'abɛrɾasˈine', 'abɛrɾatsiɔno', 'aberɾasiˈen', 'aberɾatsion', 'aberɾatsiɔeno', 'aberɾatsiˈo', 'abɛrɾatsˈoen', 'abɛrɾatsoˈine', 'abɛrɾasɔe', 'abɛrɾatsinˈ', 'abɛrɾatsieno', 'abɛrɾasoen', 'aberɾasɔoˈ', 'aberɾasˈie', 'aberɾasɔno', 'abɛrɾasiɔe', 'aberɾasin', 'aberɾatsinˈ', 'aberɾatsˈioen', 'abɛrɾasioen', 'abɛrɾasieno', 'abɛrɾatsoˈɔ', 'aberɾasiˈeno', 'aberɾatsiɔen', 'aberɾatsnoie', 'aberɾatsˈɔn', 'aberɾatsnie', 'abɛrɾatsɔeno', 'aberɾatsiˈno', 'aberɾasiɔno', 'aberɾasɔnoˈ', 'abɛrɾasoˈn', 'aberɾasoˈiɔe', 'abɛrɾatsɔe', 'abɛrɾasɔeo', 'aberɾasˈioe', 'aberɾasɔenˈ', 'abɛrɾasˈioe', 'abɛrɾatsiˈen', 'aberɾatsoˈiɔe', 'abɛrɾasnoˈie', 'aberɾatsˈioe', 'abɛrɾasnoɔ', 'abɛrɾasˈeo', 'abɛrɾasɔeno', 'aberɾasˈe', 'abɛrɾatsinoˈ', 'aberɾasˈine', 'abɛrɾatsˈn', 'aberɾasˈɔen', 'aberɾasioˈ', 'aberɾatsɔeoˈ', 'abɛrɾatsoˈɔe', 'aberɾasiɔ', 'aberɾasiˈno', 'abɛrɾasˈn', 'aberɾasioen', 'abɛrɾatsˈiɔe', 'abɛrɾatsoɔe', 'aberɾasˈeo', 'abɛrɾatsɔnoˈ', 'aberɾasˈne', 'abɛrɾasnoie', 'aberɾatsiɔno', 'abɛrɾasˈoe', 'aberɾatsinoˈ', 'aberɾatsˈno', 'aberɾasiɔeno', 'abɛrɾasoiɔ', 'abɛrɾatsn', 'aberɾasiˈeo', 'aberɾatsie', 'abɛrɾasˈo', 'abɛrɾatsnoiɔ', 'aberɾasɔe', 'abɛrɾasoiɔe', 'abɛrɾatsɔ', 'aberɾatsoe', 'aberɾasieˈ', 'abɛrɾasˈiɔ', 'abɛrɾasn', 'aberɾatsɔen', 'aberɾatsieˈ', 'abɛrɾatsne', 'abɛrɾasˈiɔn', 'abɛrɾatsˈo', 'abɛrɾasiɔ', 'abɛrɾasne', 'aberɾasɔenoˈ', 'abɛrɾasˈɔn', 'aberɾatsɔnoˈ', 'abɛrɾasinˈ', 'aberɾasˈno', 'abɛrɾatsɔo', 'abɛrɾasi', 'abɛrɾatsieo', 'abɛrɾasˈɔ', 'abɛrɾasɔno', 'abɛrɾatsˈon', 'aberɾasoiɔe', 'aberɾasi', 'abɛrɾasˈio', 'abɛrɾatsˈe', 'abɛrɾasˈioen', 'abɛrɾasiˈ', 'aberɾatsoˈin', 'abɛrɾatsiɔ', 'abɛrɾatsoˈne', 'abɛrɾatsˈion', 'abɛrɾasɔenˈ', 'aberɾatsoɔ', 'aberɾatsnoiɔ', 'abɛrɾase', 'aberɾasnɔ', 'aberɾatsoen', 'aberɾatsˈie', 'aberɾatsˈiɔn', 'abɛrɾasoˈiɔ', 'abɛrɾatsˈne', 'abɛrɾatsoie', 'aberɾatsnˈiɔ', 'aberɾasɔeoˈ', 'abɛrɾasoie', 'aberɾasiɔo', 'aberɾasoˈie', 'aberɾatsioˈ', 'abɛrɾatsɔeˈ', 'aberɾasˈo', 'aberɾasˈiɔen', 'aberɾatsienoˈ', 'abɛrɾasion', 'abɛrɾatsˈɔn', 'abɛrɾasˈoen', 'aberɾatsoine', 'aberɾase', 'abɛrɾatsˈio', 'aberɾatsiˈeo', 'abɛrɾasiˈeo', 'aberɾatsieo', 'aberɾatsɔe', 'aberɾatsˈe', 'aberɾatsˈɔ', 'aberɾatsnoɔ', 'abɛrɾatsoen', 'abɛrɾasiˈn', 'abɛrɾatsoe', 'aberɾatsoiɔe', 'abɛrɾasiɔeno', 'aberɾatsˈine', 'aberɾatsin', 'aberɾatsnoˈɔ', 'abɛrɾatsɔn', 'aberɾasˈiɔn', 'aberɾatsoˈɔe', 'aberɾasoˈiɔ', 'abɛrɾatsioen', 'aberɾatsɔn', 'abɛrɾasˈ', 'aberɾatsoie', 'abɛrɾasiˈeno', 'aberɾatsˈeo', 'abɛrɾatsˈoe', 'abɛrɾasnɔ', 'abɛrɾatsoiɔ', 'aberɾatsˈio', 'aberɾasˈiɔ', 'abɛrɾasnoe', 'abɛrɾatsiˈn', 'aberɾatsino', 'abɛrɾasio', 'abɛrɾatso', 'aberɾasoˈne', 'abɛrɾatsniɔ', 'aberɾasˈeno', 'abɛrɾasˈeno', 'abɛrɾasienˈ', 'abɛrɾatsin', 'abɛrɾatsˈno', 'aberɾasine', 'abɛrɾatsɔenˈ', 'abɛrɾatsˈ', 'abɛrɾatsiɔeno', 'aberɾatsɔˈ', 'abɛrɾasɔn', 'aberɾatsˈn', 'aberɾatsoˈn', 'abɛrɾatsnoe', 'abɛrɾasˈon', 'aberɾasɔ', 'aberɾasɔnˈ', 'aberɾatsioen', 'aberɾasˈio', 'abɛrɾasin', 'aberɾatsiɔo', 'abɛrɾatsˈiɔ', 'abɛrɾasnˈie', 'abɛrɾasiɔen', 'abɛrɾasnˈɔ', 'aberɾasnoiɔ', 'abɛrɾatsiˈe', 'abɛrɾatsienoˈ', 'aberɾatsnoˈiɔ', 'abɛrɾasɔˈ', 'abɛrɾasɔoˈ', 'abɛrɾatsɔnˈ', 'aberɾasnoe', 'abɛrɾatsˈin', 'abɛrɾatsi', 'aberɾatsiɔn', 'aberɾasoˈe']:
    if len(elt)==13:
        print(elt)
"abɛrɾatsienoˈ" "aberɾatsiˈone"