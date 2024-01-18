import torch
import torch.nn as nn
from torch import Tensor
from torchtext.vocab import Vocab

from models.types import InferenceData, InferenceData_Cognates, EOS_TOKEN, PADDING_TOKEN


def isElementOutOfRange(sequencesLengths: Tensor, maxSequenceLength: int) -> Tensor:
    """
    Return a BoolTensor with the same dimension as a tensor representing a batch of tokens sequences with variable lengths.
    False value is at the positions of closing_boundaries and padding tokens and True value at the others.
    dim = (B, L+1)

    Args:
        sequencesLengths (Tensor): a CPU IntTensor or LongTensor of B elements (B is the batch size) with the length of each batch's sequences (with boundaries).
        maxSequenceLength (int): an integer with the max sequence length (with boundaries)

    Example:
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
    """
    return torch.arange(maxSequenceLength-1).unsqueeze(0) < (sequencesLengths-1).unsqueeze(1)


def nextOneHots(targets_: InferenceData_Cognates, vocab: Vocab):
    IPA_length = len(vocab)-3
    vocSize = len(vocab)
    targets = targets_[0]
    oneHots = targets[1:].to(torch.int64)
    original_shape = oneHots.size()  # (|y|, B)
    # dim = (1, |y|, B, |Σ|+1) : the boundaries and special tokens are not interesting values for y[j] (that is why they have been erased with the reduction)
    return nn.functional.one_hot(oneHots.flatten(), num_classes=vocSize).view(original_shape+(vocSize,))[..., :IPA_length].unsqueeze(0)


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

    dim = (1, |y|+1, C, 1, hidden_dim)
    """

    targetsInputData: tuple[Tensor, Tensor]
    """
        - (IntTensor/LongTensor) The input one-hot indexes of the target cognates, without their ) closing boundary.
            dim = (|y|+1, C) ; indexes between 0 and |Σ|+1 included
        - The CPU IntTensor with the sequence lengths (with opening boundary, so |y|+1). (dim = C)
    """

    nextOneHots: Tensor
    """
    This tensor is useful to get the inferred probabilities that some phonetic tokens sub or are inserted to a current building target.
    dim = (1, |y|, C, 1, |Σ|)
    """

    maxSequenceLength: int
    """
    The maximal length of a sequence (with the boundaries) in the target cognates batch.
    """

    arePaddingElements: Tensor
    """
    dim = (C, 1, |y|+1)
    """

    lengthDataForDynProg: tuple[Tensor, int]
    """
    A tuple with:
        * the IntTensor with batch's raw sequence lengths
        * an integer equalling the maximum one
    """

    def __init__(self, targets_: InferenceData, vocab: Vocab) -> None:
        """
        Optimisation method : computes once the usefull data for the targets at the EditModel's initialisation.
        The gradient of the targets input data is set in the method to be tracked.
        """
        self.lengthDataForDynProg = (targets_[1], targets_[2])

        self.maxSequenceLength = targets_[2]+2

        targets, sequencesLengths = targets_[0], targets_[1]

        targetsInput = targets.where(
            targets != vocab[EOS_TOKEN], vocab[PADDING_TOKEN])[:-1]

        self.targetsInputData = targetsInput, targets_[1] + 1

        # dim = (1, |y|, C, 1, |Σ|) : the boundaries and special tokens are not interesting values for y[j] (that is why they have been erased with the reduction)
        self.nextOneHots = nextOneHots(
            (*self.targetsInputData, self.maxSequenceLength-1), vocab).unsqueeze(3)

        self.arePaddingElements = isElementOutOfRange(
            sequencesLengths, self.maxSequenceLength)
