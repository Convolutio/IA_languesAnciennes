from typing import Optional

import numpy as np
import numpy.typing as npt

from models.types import *
from Source.editsGraph import EditsGraph

import torch
from torch import Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# calculating the minimum edit distance between the current reconstruction
# and each of its associated cognates, we add all the strings on its minimum
# edit paths to a list, which will constitute the list of proposals for the sampling

# cf. 2.5 chapter https://web.stanford.edu/~jurafsky/slp3/2.pdf for the variables notations


def __computeMinEditDistanceMatrix(x: Tensor, y: Tensor) -> npt.NDArray[np.int_]:
    """
    We consider that a substitution has the same cost than a deletion and an insertion.
    """
    n, m = len(x), len(y)
    D = np.full((n+1, m+1), -1)  # -1 = +infinity
    D[0, :] = np.arange(m+1)
    D[1:, 0] = np.arange(1, n+1)

    for i in range(1, n+1):
        for j in range(1, m+1):
            editCosts = (D[i-1, j]+1, D[i, j-1]+1,
                         D[i-1, j-1] + (0 if x[i-1].item() == y[j-1].item() else 1))
            D[i, j] = min(editCosts)

    return D


def __getMinEditPaths(x: Tensor, y: Tensor,
                    recursivityArgs: Optional[tuple[npt.NDArray[np.int_], int, int, EditsGraph,
                                                    Optional[Edit]]] = None) -> EditsGraph:
    """
    This is all the minimal edit paths with distinct editions set. A path is modeled by a recursive
    list of edits (type Edit), which modelize an arbor.

    Args:
        x (ByteTensor): the first string to be compared (in one-hot indexes format)
        y (ByteTensor): the second one (in one-hot indexes format)
        recursivityArgs (tuple[IntMatrix, int, int, EditsGraph, Edit] | None) : (minEditDistanceMatrix, i_start, j_start, editsTree, parentEdit)
    If mentionned, this recursive function figures out the minimal edit paths between x[:i_start]
    and y[:j_start] thanks to the minEditDistanceMatrix. Else, this is the minimal edit paths\
    between x and y.
    """
    (minEditDistanceMatrix, i_start, j_start,
     editsTree, parentEdit) = None, -1, -1, None, None

    if recursivityArgs is not None:
        minEditDistanceMatrix, i_start, j_start, editsTree, parentEdit = recursivityArgs
    else:
        minEditDistanceMatrix = __computeMinEditDistanceMatrix(x, y)
        i_start = minEditDistanceMatrix.shape[0]-1
        j_start = minEditDistanceMatrix.shape[1]-1
        editsTree = EditsGraph(x, y, minEditDistanceMatrix[len(x), len(y)])

    currentPathLength = minEditDistanceMatrix[i_start, j_start]

    if currentPathLength == 0:
        return editsTree

    possiblePriorCoords: list[tuple[int, int]] = []

    if i_start > 0:
        possiblePriorCoords.append((i_start-1, j_start))
    if j_start > 0:
        possiblePriorCoords.append((i_start, j_start-1))
        if i_start > 0:
            possiblePriorCoords.append((i_start-1, j_start-1))

    minPriorPathLength = min(
        int(minEditDistanceMatrix[c[0], c[1]]) for c in possiblePriorCoords)

    possiblePriorCoords = [c for c in possiblePriorCoords
                           if int(minEditDistanceMatrix[c[0], c[1]]) == minPriorPathLength]

    if currentPathLength == minPriorPathLength:
        # reclimb the matrix with a neutral substitution
        return __getMinEditPaths(x, y, (minEditDistanceMatrix, i_start-1, j_start-1, editsTree, parentEdit))

    for c in possiblePriorCoords:
        i, j = c
        deltaCoord = (i-i_start, j-j_start)
        edit: Edit
        if deltaCoord == (-1, -1):
            # substitution
            edit = (0, i_start-1, j_start-1)
        elif deltaCoord == (0, -1):
            # insertion
            edit = (2, i_start-1, j_start-1)
        else:
            # deletion
            edit = (1, i_start-1, j_start-1)
        if not editsTree.include(edit):
            editsTree.connect(edit, parentEdit)
            __getMinEditPaths(
                x, y, (minEditDistanceMatrix, i, j, editsTree, edit))
        else:
            editsTree.connect(edit, parentEdit)

    return editsTree


class IncorrectResultsException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def __computeProposals(currentReconstruction: Tensor, cognates: list[Tensor]) -> Tensor:
    """
    Returns a list of the proposals (shape = (B, L~)) in one-hot indexes representation (sequences
    of indexes in the vocabulary)

    Args:
        currentReconstruction (ByteTensor): the current sampled proto-form
        cognates (list[ByteTensor]): its cognates\\
    """
    proposalsSet = torch.ByteTensor(size=(0, 0)).to(device)

    for cognate in cognates:
        editsTree = __getMinEditPaths(currentReconstruction, cognate)
        newComputedProposals = editsTree.computeEditsCombinations()

        if newComputedProposals.shape[1] > proposalsSet.shape[1]:
            proposalsSet = torch.nn.functional.pad(input=proposalsSet,
                                                   pad=(
                                                       0, newComputedProposals.shape[1]-proposalsSet.shape[1]),
                                                   mode='constant', value=0)
        elif newComputedProposals.shape[1] < proposalsSet.shape[1]:
            newComputedProposals = torch.nn.functional.pad(input=newComputedProposals,
                                                           pad=(
                                                               0, proposalsSet.shape[1]-newComputedProposals.shape[1]),
                                                           mode='constant', value=0)
        proposalsSet = torch.cat((proposalsSet, newComputedProposals), dim=0)

    proposalsSet = proposalsSet.unique(dim=0)
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
    return proposalsSet

def __sampleFromProposals(proposalsSet:Tensor, samplesNumber:int)->Tensor:
    """
    Arguments:
        - proposalsSet (IntTensor, shape (B, L~))
    Returns an IntTensor of shape (2*samplesNumber, L~) with random samples (uniform sampling).
    """
    proposalsNumber = len(proposalsSet)
    # drawing with 1/(n-1) law, with n the number of proposals
    j = torch.randint(high=proposalsNumber-1, size=(samplesNumber,), dtype=torch.int32, device=device)
    j = j.repeat_interleave(2) + torch.tensor([0,1], device=device).repeat(samplesNumber)
    return proposalsSet[j.unsqueeze(1), torch.arange(proposalsSet.size()[2]).unsqueeze(0)]

def generateProposalsFromCurrentReconstructions(currentReconstructions: list[Tensor], cognates: list[dict[ModernLanguages, Tensor]], samplesNumber:int) -> list[Tensor]:
    """
    For each cognate pair, generate a list of proposals from the current chosen sample and its corresponding cognates in each languages. The list of proposals processed by the MH algorithms
    is then established with a uniform random drawing of `samplesNumber` items.
    Then a list of c IntTensors of shape (`samplesNumber*2`, L~) is generated.
    """
    p = list()
    numberOfCognatePairs = len(cognates)
    print('-'*60)
    for i in range(numberOfCognatePairs):
        x = currentReconstructions[i]
        Y = list(cognates[i].values())
        p.append(__sampleFromProposals(__computeProposals(x, Y), samplesNumber))
        if (i+1) % (numberOfCognatePairs//100) == 0:
            print("Proposals generation:", 1+100*(i+1) //
                  numberOfCognatePairs, '%'+' '*10, end='\r')
    print('\n'+'-'*60+'\n')
    return p
