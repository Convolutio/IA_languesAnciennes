#!/usr/local/bin/python3.9.10

import torch
import torch.nn as nn

from torch import Tensor, device
from torch.types import Device

from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.functional import pad

from torchtext.vocab import Vocab

from ..models.types import (ModernLanguages, Operations,
                          InferenceData_SamplesEmbeddings, InferenceData_Cognates, PADDING_TOKEN)
from .packingEmbedding import PackingEmbedding


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

    def __init__(self, language: ModernLanguages, shared_embedding_layer: PackingEmbedding, vocab: Vocab, device: Device, lstm_hidden_dim: int):
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

        super(EditModel, self).__init__()
        
        # To handle correctly the type of `device` (which is *Device*) is turn into a *str*, then convert by the function into a type of *device*
        self.device = torch.device(device=f"{device}")
        self.language = language

        IPA_length = len(vocab)-3
        # must equals |Σ|+1 for the <end>/<del> special output tokens
        self.output_dim = IPA_length + 1
        self.padding_index = vocab[PADDING_TOKEN]
        lstm_input_dim = shared_embedding_layer.embedding_dim

        self.encoder_prior = nn.Sequential(
            nn.LSTM(lstm_input_dim, lstm_hidden_dim//2, bidirectional=True)
        ).to(self.device)
        self.encoder_modern = nn.Sequential(
            shared_embedding_layer,
            nn.LSTM(lstm_input_dim, lstm_hidden_dim)
        ).to(self.device)

        self.sub_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, self.output_dim),
            nn.LogSoftmax(dim=-1)
        ).to(self.device)
        self.ins_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, self.output_dim),
            nn.LogSoftmax(dim=-1)
        ).to(self.device)

        self.__cachedProbs: dict[Operations, Tensor] = {}

    def __call__(self, sources_: InferenceData_SamplesEmbeddings, targets_: InferenceData_Cognates) -> tuple[Tensor, Tensor]:
        return super().__call__(sources_, targets_)

    def forward(self, sources_: InferenceData_SamplesEmbeddings, targets_: InferenceData_Cognates) -> tuple[Tensor, Tensor]:
        """
        Returns a tuple of tensors representing respectively log q_sub(. |x,.,y[:.]) and log q_ins(.|x,.,y[:.]) , for each (x,y) sample-target couple in the batch.

        Tensors dimension = (|x|+1, |y|+1, C, B, |Σ|+1)

        The value for padded elements are not neutralized. Please, apply a padding mask to the results if needed.

        Args :
            sources_
            targets_ : if not specified (with None, in inference stage), the context will be got from the cached data.
        """
        # shape = (1, |y|+1, C, 1, hidden_dim)
        modernContext = self.encoder_modern(
            (targets_[0], targets_[1], False))[0]
        modernContext = pad_packed_sequence(
            modernContext)[0].unsqueeze(-2).unsqueeze(0)

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

    def forward_and_select(self, sources: InferenceData_SamplesEmbeddings, targets: InferenceData_Cognates) -> dict[Operations, Tensor]:
        """
        Run the forward method of the model and select the probabilities of interest for the cognates provided in input. During the selection, the probabilities for undefined (x[i],y[:j]) input couples (i.e. with one of the two input containing at least one padding token) are neutralized.
        Returns a dictionnary of the probabilities for each operations (ins, sub, end, dlt).

        Tensors shape = (|x|+1, |y|+1, c, b)
        """
        def convertForIndexing(t): return torch.cat(
            (t, torch.zeros((*t.size()[:-1], 1), device=self.device)), dim=-1)

        sub_results, ins_results = (convertForIndexing(t)
                                    for t in self(sources, targets))
        x_l, y_l, c, b = sub_results.shape[:-1]  # |x|+1, |y|+1, c, b

        # neutralizes results for the padding and eos tokens
        not_padding_token_in_source = (torch.arange(sources[2]-1)[:, None, None] < (
            sources[1]-1).unsqueeze(0)).unsqueeze(1).unsqueeze(-1).to(self.device) # shape = (|x|+1, 1, c, b, 1)

        cognates = targets[0].detach()  # shape = (|y|+1, c)

        targetsCoords = pad(cognates[1:], (0, 0, 0, 1), "constant",
                            self.output_dim).unsqueeze(-1)  # shape = (|y|+1, c, 1)
        targetsCoords = targetsCoords.where(
            targetsCoords != self.padding_index, self.output_dim)
        targetsCoords = torch.cat((targetsCoords,
                                   torch.where(cognates != self.padding_index, self.output_dim - 1, self.output_dim).unsqueeze(-1)),
                                  dim=-1)  # shape = (|y|+1, c, 2)
        targetsCoords = targetsCoords.unsqueeze(
            0).unsqueeze(3)  # shape = (1, |y|+1, c, 1, 2)
        targetsCoords = torch.where(not_padding_token_in_source,
                                    targetsCoords, self.output_dim)
        targetsCoords = torch.meshgrid(torch.arange(x_l),
                                       torch.arange(y_l),
                                       torch.arange(c),
                                       torch.arange(b),
                                       torch.arange(2),
                                       indexing='ij'
                                       )[:-1] + (targetsCoords,)

        # shape (|x|+1, |y|+1, c, b, 2)
        sub_results = sub_results[targetsCoords]
        # shape (|x|+1, |y|+1, c, b, 2)
        ins_results = ins_results[targetsCoords]

        return {'sub': sub_results[..., 0], 'dlt': sub_results[..., 1],
                'ins': ins_results[..., 0], 'end': sub_results[..., 1]}

    def cache_probs(self, sources: InferenceData_SamplesEmbeddings, targets: InferenceData_Cognates):
        """
        Runs inferences in the model from given sources. 
        It is supposed that the context of the targets and their one-hots have already been computed in the model.
        """
        self.eval()
        with torch.no_grad():

            def lengthen(t, padding_x, padding_y):
                return torch.cat((torch.cat((t, torch.zeros((padding_x, *t.size()[1:]), device=self.device)), dim=0),
                                  torch.zeros((t.size()[0]+1, padding_y, *t.size()[2:]), device=self.device)), dim=1)

            # dim = (|x|+2, |y|+2, c, b)
            self.__cachedProbs = {op: lengthen(t, 1, 1)
                                  for (op, t) in self.forward_and_select(sources, targets).items()}

    def clear_cache(self):
        self.__cachedProbs = {}

    def ins(self, i: int, j: int):
        return self.__cachedProbs['ins'][i, j]

    def sub(self, i: int, j: int):
        return self.__cachedProbs['sub'][i, j]

    def end(self, i: int, j: int):
        return self.__cachedProbs['end'][i, j]

    def dlt(self, i: int, j: int):
        return self.__cachedProbs['dlt'][i, j]
