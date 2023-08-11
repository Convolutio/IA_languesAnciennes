#!/usr/local/bin/python3.9.10
from typing import Optional, Literal, Callable

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence
from torchtext.vocab import Vocab
from torch import Tensor

from models.probcache import ProbCache
from models.articleModels import *
from models.models import InferenceData, SourceInferenceData, TargetInferenceData, SOS_TOKEN, EOS_TOKEN, PADDING_TOKEN
from source.packingEmbedding import PackingEmbedding
from source.cachedModernData import CachedTargetsData, isElementOutOfRange

device = "cuda" if torch.cuda.is_available() else "cpu"

BIG_NEG = -1e9


class EditModel(nn.Module):
    """
    # Neural Edit Model

    ### This class gathers the neural insertion and substitution models, specific to a branch between the\
          proto-language and a modern language.
    `q_ins` and `q_sub` are defined as followed (considering Σ as the IPA characters vocabulary):
        * Reconstruction input (samples): a batch of packed sequences of embeddings representing phonetic or special tokens\
        in Σ ∪ {`'('`, `')'`, `'<pad>'`}
        * Targets input (cognates): a batch of padded sequences of one-hot indexes from 0 to |Σ|+1 representing phonetic or special tokens in Σ ∪ {`'('`, `'<pad>'`}. The processing of the closing boundary at the end of the sequence will be avoided by the model thanks to the unidirectionnality of the context. 
        * Output: a tensor of dimension |Σ|+1 representing a probability distribution over\
        Σ ∪ {`'<del>'`} for `q_sub` or Σ ∪ {`'<end>'`} for `q_ins`. The usefull output batch has a dimension of (|x|+1, |y|+1, B)

    ### Instructions for targets' data caching
        * At the beginning of the Monte-Carlo training, cache the initial targets' background data by adding an InferenceData object in __init__'s argument
        * At the beginning of each EM iteration, run the inference of the targets' context once with the `update_cachedTargetContext` method, so the context will be computed once for all the sampling step and not at each MH sampling iteration.

    ### Inference in the model
    With the `cache_probs` method. Then, use the `sub`, `ins`, `del` and `end` method to get the cached probs.

    ### Training
    Compute the logits with mini batches of samples and cognates input data (so the cache will not be used here).

    Expected input samples tensor shape (in a PackedSequence): (|x|+2, C*B, input_dim)\\
    Expected input samples' lengths tensor shape: (C, B)

    Expected input cognates tensor shape: (|y|+1, C)
    Expected input cognates' lengths tensor shape: (C)
    """

    def __init__(self, targets_: InferenceData, language: ModernLanguages, shared_embedding_layer: PackingEmbedding, vocab: Vocab, lstm_hidden_dim: int):
        """
        Params:
            - embedding_layer (Embedding): the shared embedding layer between all of the EditModels. It hosts an input containing tokens in the vocabulary passed in the 'vocab' argument object.
            The embeddings will be passed in input of the lstm models.
            - voc_size (int): the size of the IPA characters vocabulary (without special tokens), to compute the dimension of the one-hot vectors that the model will have to host in input, but to also figure out the number of classes in the output distribution.
            - lstm_hidden_dim (int): the arbitrary dimension of the hidden layer computed by the lstm models, the unidirectional as well as the bidirectional. Therefore, this dimension must be an even integer. 
        """
        assert (lstm_hidden_dim % 2 ==
                0), "The dimension of the lstm models' hidden layer must be an even integer"
        assert (shared_embedding_layer.num_embeddings == len(vocab)
                ), "The shared embedding layer has been wrongly set."

        self.language = language

        super(EditModel, self).__init__()

        IPA_length = len(vocab)-3
        # must equals |Σ|+1 for the <end>/<del> special output tokens
        self.output_dim = IPA_length + 1
        lstm_input_dim = shared_embedding_layer.embedding_dim

        self.encoder_prior = nn.Sequential(
            nn.LSTM(lstm_input_dim, lstm_hidden_dim//2, bidirectional=True)
        ).to(device)
        self.encoder_modern = nn.Sequential(
            shared_embedding_layer,
            nn.LSTM(lstm_input_dim, lstm_hidden_dim)
        ).to(device)

        self.sub_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, self.output_dim),
            nn.LogSoftmax(dim=-1)
        ).to(device)
        self.ins_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, self.output_dim),
            nn.LogSoftmax(dim=-1)
        ).to(device)

        self.__cachedTargetsData = CachedTargetsData(targets_, vocab)

        self.__cachedProbs: dict[Literal['ins',
                                         'sub', 'end', 'del'], Tensor] = {}

    @property
    def cachedTargetInputData(self):
        """
        * (IntTensor/LongTensor) The input one-hot indexes of the target cognates, without their ) closing boundary. dim = (|y|+1, B) ; indexes between 0 and |Σ|+1 included
        * The CPU IntTensor with the sequence lengths (with opening boundary, so |y|+1). (dim = B)
        """
        return self.__cachedTargetsData.targetsInputData

    @property
    def cachedTargetDataForDynProg(self):
        """
        A tuple with:
            * the ndarray with batch's raw sequence lengths
            * an integer equalling the maximum one
        """
        return self.__cachedTargetsData.lengthDataForDynProg

    # ----------Inference---------------

    def update_cachedTargetContext(self):
        """
        Before each inference stage (during the sampling and the backward dynamic program running), call this method to cache the current inferred context for the modern forms.
        """
        targetsInputData = self.__cachedTargetsData.targetsInputData
        self.__cachedTargetsData.modernContext = pad_packed_sequence(
            self.encoder_modern((*targetsInputData, False))[0])[0].unsqueeze(-2).unsqueeze(0)

    def __call__(self, sources_: SourceInferenceData, targets_: Optional[TargetInferenceData] = None) -> tuple[Tensor, Tensor]:
        return super().__call__(sources_, targets_)

    def forward(self, sources_: SourceInferenceData, targets_: Optional[TargetInferenceData]) -> tuple[Tensor, Tensor]:
        """
        Returns a tuple of tensors representing respectively log q_sub(. |x,.,y[:.]) and log q_ins(.|x,.,y[:.]) , for each (x,y) sample-target couple in the batch.

        Tensors dimension = (|x|+1, |y|+1, C, B, |Σ|+1)

        The value for padded elements are not neutralized. Please, apply a padding mask to the results if needed.

        Args :
            sources_
            targets_ : if not specified (with None, in inference stage), the context will be got from the cached data.
        """
        modernContext: Tensor  # dim = (1, |y|+1, C, 1, hidden_dim)
        if targets_ is not None:
            modernContext = pad_packed_sequence(self.encoder_modern(
                (targets_[0], targets_[1], False))[0])[0].unsqueeze(-2).unsqueeze(0)
        else:
            modernContext = self.__cachedTargetsData.modernContext

        cognatePairsNumber = modernContext.size()[2]

        priorContext = pad_packed_sequence(self.encoder_prior(sources_[0])[0])[
            0][:-1]  # shape = (|x|+1, C*B, hidden_dim)
        currentShape = priorContext.size()
        priorContext = priorContext.view(currentShape[0],
                                         cognatePairsNumber, currentShape[1]//cognatePairsNumber,
                                         currentShape[2]
                                         ).unsqueeze(1)  # dim = (|x|+1, 1, C, B, hidden_dim)

        # dim = (|x|+1, |y|+1, C, B, hidden_dim)
        ctx = priorContext + modernContext

        # dim = (|x|+1, |y|+1, C, B, |Σ|+1)
        sub_results: Tensor = self.sub_head(ctx)
        # dim = (|x|+1, |y|+1, C, B, |Σ|+1)
        ins_results: Tensor = self.ins_head(ctx)

        return sub_results, ins_results

    def cache_probs(self, sources: list[SourceInferenceData]):
        """
        Runs inferences in the model from given sources. 
        It is supposed that the context of the targets and their one-hots have already been computed in the model.
        """
        self.eval()
        with torch.no_grad():
            def convertForIndexing(t): return torch.cat(
                (t, torch.zeros((*t.size()[:-1], 1), device=device)), dim=-1)

            def lengthen(t, padding_x, padding_y): return torch.cat((
                torch.cat(
                    (t, torch.zeros((padding_x, *t.size()[1:]), device=device)), dim=0),
                torch.zeros((t.size()[0]+1, padding_y,
                            *t.size()[2:]), device=device)
            ), dim=1)

            # dim = (|x|+2, |y|+2, C, B)
            self.__cachedProbs = {}

            cachedProbsNotConcatenated: dict[Literal['ins', 'sub', 'end', 'del'], list[Tensor]] = {
                key: [] for key in ('ins', 'sub', 'end', 'del')}

            # TODO: enhance performances with memory usage and parallelisation
            for miniSources in sources:
                sub_results, ins_results = (
                    convertForIndexing(t) for t in self(miniSources))
                x_l, y_l, C, b = sub_results.shape[:-1]  # |x|+1, |y|+1, C, b

                # usefull space = (|x|+1, |y|+1, C, b)
                cachedProbsNotConcatenated['del'] += lengthen(
                    sub_results[..., -2], 1, 1).chunk(b, -1)
                # usefull space = (|x|+1, |y|+1, C, b)
                cachedProbsNotConcatenated['end'] += lengthen(
                    ins_results[..., -2], 1, 1).chunk(b, -1)

                neutralizeOutOfRangeClasses: Callable[[Tensor], Tensor] = lambda t: t.where(
                    t < self.output_dim, self.output_dim)
                targetsOneHots = neutralizeOutOfRangeClasses(
                    self.__cachedTargetsData.targetsInputData[0][1:])  # shape = (|y|, C)
                indexes = torch.meshgrid(torch.arange(x_l), torch.arange(y_l-1), torch.arange(
                    C), torch.arange(b), indexing='ij') + (targetsOneHots.unsqueeze(0).unsqueeze(-1),)

                # q(y[j]| x, i, y[:j]) = q(.| x, i, y[:j])[onehot_idx(y[j])]
                # usefull space = (|x|+1, |y|, C, B)
                cachedProbsNotConcatenated['sub'] += lengthen(
                    sub_results[indexes], 1, 2).chunk(b, -1)
                # usefull space = (|x|+1, |y|, C, B)
                cachedProbsNotConcatenated['ins'] += lengthen(
                    ins_results[indexes], 1, 2).chunk(b, -1)

            while len(cachedProbsNotConcatenated) > 0:
                operation, tensorLst = cachedProbsNotConcatenated.popitem()
                self.__cachedProbs[operation] = pad_sequence(tensorLst).transpose(
                    1, -1).squeeze(1)  # shape = (|x|+2, |y|+2, C, B)

    def clear_cache(self):
        self.__cachedProbs = {}

    def ins(self, i: int, j: int):
        return self.__cachedProbs['ins'][i, j]

    def sub(self, i: int, j: int):
        return self.__cachedProbs['sub'][i, j]

    def end(self, i: int, j: int):
        return self.__cachedProbs['end'][i, j]

    def dlt(self, i: int, j: int):
        return self.__cachedProbs['del'][i, j]

    # ----------Training----------------
    def __computePaddingMask(self, sourceLengthData: tuple[Tensor, int], targetLengthData: tuple[Tensor, int]):
        """
        Computes a mask for the padded elements from the reconstructions x and the targets y.

        Mask dim = (|x|+1, |y|+1, b, 1)

        Apply the mask with broadcasting as below
        >>> mask = self.__computeMask(sourcesLengths, maxSourceLength)
        >>> masked_ctx = mask*ctx

        Args:
            * sourceLengthData: a tuple with the following elements:
                - sourcesLengths: a cpu tensor containing the lengths of the samples with the boundaries (|x|+2)
                - maxSourceLength: the value of the maximum sequence length (including the boundaries).
            * targetLengthData: if None, this data is not computed and got from the cache. Else, the tuple with the following elements must be given:
                - targetsLengths: a cpu tensor containing the lengths of the modern forms with the boundaries (|y|+2)
                - maxTargetLength: the value of the maximum sequence length (including the boundaries)
        """
        A = isElementOutOfRange(*sourceLengthData)  # dim = (b, |x|+1)
        B = isElementOutOfRange(*targetLengthData)  # dim = (b, |y|+1)
        # dim = (|x|+1, |y|+1, b, 1)
        return torch.logical_and(A.unsqueeze(-1), B.unsqueeze(-2)).to(device).movedim((1, 2), (0, 1)).unsqueeze(-1)

    def __computeMaskedProbDistribs(self, sub_probs: Tensor, ins_probs: Tensor, targetOneHotVectors: Tensor) -> Tensor:
        """
        Returns a tensor of shape (|x|+1, |y|+1, b, 2*(|Σ|+1)) with a mask applied on each probability in the distribution which are not defined in the target probabilities cache. This is done according to the one-hots of the characters of interest in the target batch.

        Args:
            - sub_probs (Tensor): the logits or the targets for q_sub probs, with at the end the deletion prob
                dim = (|x|+1, |y|+1, b, |Σ|+1) or (|x|+1, |y|+1, b, 2) 
            - ins_probs (Tensor): the logits or the targets for q_ins probs, with at the end the insertion prob
                dim = (|x|+1, |y|+1, b, |Σ|+1) or (|x|+1, |y|+1, b, 2)
            - targetOneHotVectors (Tensor): dim = (1, |y|, b, |Σ|), the one hot vectors of the IPA characters in the modern forms mini batch.
            It is assumed that the padding mask has already been applied on the probs in argument.
        """
        b, V = targetOneHotVectors.shape[-2:]
        nextOneHots = torch.cat(
            (targetOneHotVectors, torch.zeros((1, 1, b, V), device=device)), dim=1)
        masked_sub_probs = torch.nan_to_num(
            nextOneHots * sub_probs[:, :, :, :-1], nan=0.)
        masked_ins_probs = torch.nan_to_num(
            nextOneHots * ins_probs[:, :, :, :-1], nan=0.)
        deletionProbs = sub_probs[:, :, :, -1:]
        endingProbs = ins_probs[:, :, :, -1:]
        return torch.cat((masked_sub_probs, deletionProbs, masked_ins_probs, endingProbs), dim=-1)

    def get_targets(self, prob_targets_cache: ProbCache, samplesLengthData: tuple[Tensor, int], targetsLengthData: tuple[Tensor, int], targetOneHotVectors: Tensor):
        """
        This method computes the targets probs tensor with a good format and an applied padding mask, which is computed thanks to the samples' lengths IntTensor.

        Targets distrib tensor shape = (|x|+1, |y|+1, b, 2*(|Σ|+1))

        Args:
            * prob_targets_cache (ProbCache) : the targets output probabilities computed from the backward dynamic program.
            * samplesLengthData (IntTensor, int): the CPU IntTensor/LongTensor (shape = (b)) with the samples' sequence lengths (with boundaries, so |x|+2) and the integer with the maximum one. It is expected data for the whole batch to be given.
            * targetsLengthData (IntTensor, int): the CPU IntTensor/LongTensor (shape = (b)) with the targets' sequence lengths (with boundaries, so |y|+2) and the integer with the maximum one. It is expected data for the whole batch to be given.
            * targetOneHotVectors (Tensor): dim = (1, |y|, b, |Σ|), the one hot vectors of the IPA characters in the modern forms mini batch.

            It is assumed that the padding mask has already been applied on the probs in argument.
        """
        with torch.no_grad():
            paddingMask = self.__computePaddingMask(
                samplesLengthData, targetsLengthData)

            prob_targets = self.__computeMaskedProbDistribs(
                torch.nan_to_num(paddingMask*torch.cat((
                    prob_targets_cache.sub[:-1, :-1],
                    prob_targets_cache.dlt[:-1, :-1]
                ), dim=-1), nan=0.),
                torch.nan_to_num(paddingMask*torch.cat((
                    prob_targets_cache.ins[:-1, :-1],
                    prob_targets_cache.end[:-1, :-1]
                ), dim=-1), nan=0.),
                targetOneHotVectors
            )
        return prob_targets

    def get_logits(self, samples_: SourceInferenceData, targets_: TargetInferenceData, targetOneHotVectors: Tensor):
        """
        In training mode, computes the logits in a training step, from the specified samples and targets minibatch.
        logits tensor shape: (|x|+1 , |y|+1 , b, 2*(|Σ|+1)).
        """

        flat: Callable[[Tensor], Tensor] = lambda t: t.squeeze(-2)
        sub_outputs, ins_outputs = [flat(t) for t in self(samples_, targets_)]
        paddingMask = self.__computePaddingMask((samples_[1].squeeze(-1), samples_[2]),
                                                (targets_[1]+1, targets_[2]+1))
        masked_outputs = self.__computeMaskedProbDistribs(
            sub_outputs*paddingMask, ins_outputs*paddingMask, targetOneHotVectors)

        return masked_outputs
