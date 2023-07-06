from torch import Tensor
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn as nn
import torch

import numpy as np

from Types.models import InferenceData, TargetInferenceData

def isElementOutOfRange(sequencesLengths:Tensor, maxSequenceLength:int) -> Tensor:
    """
    Return a BoolTensor with the same dimension as a tensor representing a batch of tokens sequences with variable lengths.
    False value is at the positions of closing_boundaries and padding tokens and True value at the others.
    dim = (B, L+1)

    ## Example:

    4 and 5 are the respectives one-hot indexes for ( and )
    >>> batch = torch.tensor([
        [4,1,1,2,5,0],
        [4,1,3,5,0,0],
        [4,2,2,3,5,0],
        [4,3,5,0,0,0]])
    >>> batch_sequencesLengths = torch.tensor([3,2,3,1], dtype=int) + 2
    >>> maxSequenceLength = 3+2 # torch.max(batch_sequencesLengths)
    >>> isElementOutOfRange(batch_sequencesLengths, maxSequenceLength)
    tensor([[True, True, True, True, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, False, False, False, False]])

    ## Arguments:
        - sequencesLengths: a CPU IntTensor or LongTensor of B elements (B is the batch size) with the length of each batch's sequences (with boundaries).
        - maxSequenceLength: an integer with the max sequence length (with boundaries)
    """
    return torch.arange(maxSequenceLength-1).unsqueeze(0) < (sequencesLengths-1).unsqueeze(1)

def nextOneHots(targets_:TargetInferenceData, voc_size:int):
    targets, sequencesLengths = targets_[:2]
    oneHots = torch.where(targets == 0, 0, targets - 1)[1:].to(torch.int64)
    original_shape = oneHots.size() # (|y|, B)
    # dim = (1, |y|, B, |Σ|+1) : the boundaries and special tokens are not interesting values for y[j] (that is why they have been erased with the reduction)
    return pad_packed_sequence(pack_padded_sequence(nn.functional.one_hot(oneHots.flatten(), num_classes=voc_size).view(original_shape+(voc_size,)), sequencesLengths-1, False, False), False)[0].unsqueeze(0)

class CachedTargetsData():
    """
    Contains :
        * modernContext (Tensor) : the context embeddings computed from the targets
        * targetsInputData (Tensor) : 
        * sequencesLengths (ByteTensor) : the lengths of each sequence
        * maxSequenceLength (int)
        * arePaddingElements(BoolTensor) : A coefficient equals True if its position is not in a padding zone
    """
    modernContext: Tensor
    """
    Context of y, ready to be summed with the context of x.

    dim = (B, 2*hidden_dim, 1, |y|+1)
    """

    targetsInputData: tuple[Tensor, Tensor]
    """
        - (IntTensor/LongTensor) The input one-hot indexes of the target cognates, without their ) closing boundary.
            dim = (|y|+1, B) ; indexes between 0 and |Σ|+1 included
        - The CPU IntTensor with the sequence lengths (with opening boundary, so |y|+1). (dim = B)
    """
    
    nextOneHots: Tensor
    """
    This tensor is useful to get the inferred probabilities that some phonetic tokens sub or are inserted to a current building target.
    dim = (1, |y|, B, |Σ|)
    """

    maxSequenceLength: int
    """
    The maximal length of a sequence (with the boundaries) in the target cognates batch.
    """
    
    arePaddingElements: Tensor
    """
    dim = (B, |y|+1)
    """

    lengthDataForDynProg: tuple[np.ndarray, int]
    """
    A tuple with:
        * the ndarray with batch's raw sequence lengths
        * an integer equalling the maximum one
    """

    def __init__(self, targets_:InferenceData) -> None:
        """
        Optimisation method : computes once the usefull data for the targets at the EditModel's initialisation.
        The gradient of the targets input data is set in the method to be tracked.
        """
        self.lengthDataForDynProg = (targets_[1].numpy(), targets_[2])

        self.maxSequenceLength = targets_[2]+2
        
        targets, sequencesLengths = targets_[0], targets_[1]

        closing_boundary_index = int(torch.max(targets).item())
        voc_size = closing_boundary_index - 2
        targetsInput = targets.where(targets != closing_boundary_index, 0)[:-1]
        
        self.targetsInputData = targetsInput, targets_[1] + 1
        
        # dim = (1, |y|, B, |Σ|) : the boundaries and special tokens are not interesting values for y[j] (that is why they have been erased with the reduction)
        self.nextOneHots = nextOneHots((*self.targetsInputData, self.maxSequenceLength-1), voc_size)
        
        self.arePaddingElements = isElementOutOfRange(sequencesLengths, self.maxSequenceLength)