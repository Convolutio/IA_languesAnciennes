from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence


class PackingEmbedding(nn.Embedding):
    """
    This module computes the embeddings for sequences of tensor with one-hot indexes in fixed vocabulary and after packs the embeddings sequences.

    Forward Arguments (all in the same tuple):
        - inputTensor (Tensor): a batch which contains sequences of tokens, represented with one-hot indexes.
        - lengths (IntTensor, CPU): a list of each sequence's length of interest.
        - batch_first (bool): if True, the inputTensor shape must be (B, L), else (L, B)
    """
    def forward(self, inpt:tuple[Tensor, Tensor, bool]):
        inputTensor, lengths, batch_first = inpt
        embedded = super().forward(inputTensor)
        packed = pack_padded_sequence(embedded, lengths, batch_first, enforce_sorted=False)
        return packed
    
    def __call__(self, inputTensor:Tensor, lengths:Tensor, batch_first:bool) -> PackedSequence:
        """
        Forward Arguments (all in the same tuple):
        - inputTensor (Tensor): a batch which contains sequences of tokens, represented with one-hot indexes.
        - lengths (IntTensor, CPU): a list of each sequence's length of interest.
        - batch_first (bool): if True, the inputTensor shape must be (B, L), else (L, B)
        """
        return super().__call__((inputTensor, lengths, batch_first))
