#!/usr/local/bin/python3.9.10
from typing import Optional

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence
from torch import Tensor

from models import ProbCache
from Types.articleModels import *
from Types.models import InferenceData, SourceInferenceData, TargetInferenceData
from Source.packingEmbedding import PackingEmbedding
from Source.cachedModernData import CachedTargetsData, isElementOutOfRange

device = "cuda" if torch.cuda.is_available() else "cpu"

BIG_NEG = -1e9


class EditModel(nn.Module):
    """
    # Neural Edit Model

    ### This class gathers the neural insertion and substitution models, specific to a branch between the\
          proto-language and a modern language.
    `q_ins` and `q_sub` are defined as followed:
        * Reconstruction input (samples): a batch of packed sequences of embeddings representing phonetic or special tokens\
        in Σ ∪ {`'('`, `')'`, `'<pad>'`}
        * Targets input (cognates): a batch of padded sequences of one-hot indexes from 0 to |Σ|+1 representing phonetic or special tokens in Σ ∪ {`'('`, `')'`}. The processing of the closing boundary at the end of the sequence will be avoided by the model thanks to the unidirectionnality of the context. 
        * Output: a tensor of dimension |Σ|+1 representing a probability distribution over\
        Σ ∪ {`'<del>'`} for `q_sub` or Σ ∪ {`'<end>'`} for `q_ins`. The usefull output batch has a dimension of (|x|+1, |y|+1, B) 

    ### Instructions for targets' data caching
        * At the beginning of the Monte-Carlo training, cache the initial targets' background data by adding an InferenceData object in __init__'s argument
        * At the beginning of each EM iteration, run the inference of the targets' context once with the `update_cachedTargetContext` method, so the context will be computed once for all the sampling step and not at each MH sampling iteration.
    
    ### Inference in the model
    With the `cache_probs` method. Then, use the `sub`, `ins`, `del` and `end` method to get the cached probs.

    #### About the padding
    In the cached probs, some probabilities are undefined and has the neutral logarithmic probability equaling 0.

    ### Training
    TODO: correctly defining it
    """

    def __init__(self, targets_:InferenceData, language: ModernLanguages, shared_embedding_layer:PackingEmbedding, voc_size:int, lstm_hidden_dim: int):
        """
        Arguments:
            - embedding_layer (Embedding): the shared embedding layer between all of the EditModels. It hosts an input containing one-hot indexes between 0 and voc_size+2 included (with 

                        - 0 the empty padding token (must be ignored with the padding_idx=0 parameter setting),
                        - voc_size + 1 the `(` opening boundary token,
                        - voc_size + 2 the `)` closing boundary token
                
                )
            The embeddings will be passed in input of the lstm models.
            - voc_size (int): the size of the IPA characters vocabulary (without special tokens), to compute the dimension of the one-hot vectors that the model will have to host in input, but to also figure out the number of classes in the output distribution.
            - lstm_hidden_dim (int): the arbitrary dimension of the hidden layer computed by the lstm models, the unidirectional as well as the bidirectional. Therefore, this dimension must be an even integer. 
        """
        assert(lstm_hidden_dim % 2 ==0), "The dimension of the lstm models' hidden layer must be an even integer"
        assert(shared_embedding_layer.padding_idx==0 and shared_embedding_layer.num_embeddings==voc_size+3), "The shared embedding layer has been wrongly set."

        self.language = language
        
        super(EditModel, self).__init__()
        
        self.output_dim = voc_size + 1  # must equals |Σ|+1 for the <end>/<del> special output tokens
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

        self.__cachedTargetsData = CachedTargetsData(targets_)
        
        self.__cachedProbs: dict[Literal['ins',
                                         'sub', 'end', 'del'], np.ndarray] = {}
        
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
    
    #----------Inference---------------

    def update_cachedTargetContext(self):
        """
        Before each inference stage (during the sampling and the backward dynamic program running), call this method to cache the current inferred context for the modern forms.
        """
        targetsInputData = self.__cachedTargetsData.targetsInputData
        self.__cachedTargetsData.modernContext = pad_packed_sequence(self.encoder_modern((*targetsInputData, False))[0])[0]

    def __computePaddingMask(self, sourceLengthData:tuple[Tensor, int], targetLengthData:Optional[tuple[Tensor, int]]):
        """
        Computes a mask for the padded elements from the reconstructions x and the targets y.
        
        Mask dim = (|x|+1, |y|+1, B, 1)

        Apply the mask with broadcasting as below
        >>> mask = self.__computeMask(sourcesLengths, maxSourceLength)
        >>> masked_ctx = mask*ctx

        Arguments:
            * sourceLengthData: a tuple with the following elements:
                - sourcesLengths: a cpu tensor containing the lengths of the samples with the boundaries (|x|+2)
                - maxSourceLength: the value of the maximum sequence length (including the boundaries).
            * targetLengthData: if None, this data is not computed and got from the cache. Else, the tuple with the following elements must be given:
                - targetsLengths: a cpu tensor containing the lengths of the modern forms with the boundaries (|y|+2)
                - maxTargetLength: the value of the maximum sequence length (including the boundaries)
        """
        A = isElementOutOfRange(*sourceLengthData) # dim = (B, |x|+1)
        B = self.__cachedTargetsData.arePaddingElements # dim = (B, |y|+1)
        if targetLengthData is not None:
            B = isElementOutOfRange(*targetLengthData) # dim = (B, |y|+1)
        return torch.logical_and(A.unsqueeze(-1), B.unsqueeze(-2)).to(device).movedim((1, 2), (0, 1)).unsqueeze(-1) # dim = (|x|+1, |y|+1, B, 1)
    
    
    def __call__(self, sources_:SourceInferenceData, targets_:Optional[TargetInferenceData] = None) -> tuple[Tensor, Tensor]:
        return super().__call__(sources_, targets_)
    
    def forward(self, sources_:SourceInferenceData, targets_:Optional[TargetInferenceData]) -> tuple[Tensor, Tensor]:
        """
        Returns a tuple of tensors representing respectively log q_sub(. |x,.,y[:.]) and log q_ins(.|x,.,y[:.]) , for each (x,y) sample-target couple in the batch.

        Tensors dimension = (|x|+1, |y|+1, B, |Σ|+1)
        
        The value for padded sequence elements is 0, which infers an undefined and neutral probability distribution for these elements. Moreover, if log(logit) = log(target) = 0, so the contribution of padded tokens in the chosen loss is neutralized.
        
        Arguments :
            - sources_ : (sources, lengths, maxSequenceLength), with sources a ByteTensor of dim = (|x|+2, B)
            - targets_ : if not specified (with None, in inference stage), the context and mask will be got from the cached data.
        """
        priorContext, sourcesLengths = pad_packed_sequence(self.encoder_prior(sources_[0])[0])
        priorMaxSequenceLength = priorContext.size()[0] # |x|+2
        priorContext = priorContext[:-1] # dim = (|x|+1, B, hidden_dim)

        modernContext:Tensor # dim = (|y|+1, B, hidden_dim)
        if targets_ is not None:
            modernContext = pad_packed_sequence(self.encoder_modern((targets_[0], targets_[1], False))[0])[0]
        else:
            modernContext = self.__cachedTargetsData.modernContext

        ctx = priorContext.unsqueeze(1) + modernContext.unsqueeze(0) # dim = (|x|+1, |y|+1, B, hidden_dim)
        
        mask = self.__computePaddingMask((sourcesLengths, priorMaxSequenceLength),
                                         (targets_[1]+1, targets_[2]+1) if targets_ is not None else None)
        masked_ctx = mask*ctx

        sub_results:Tensor = self.sub_head(masked_ctx)*mask  # dim = (|x|+1, |y|+1, B, |Σ|+1)
        ins_results:Tensor = self.ins_head(masked_ctx)*mask  # dim = (|x|+1, |y|+1, B, |Σ|+1)
        
        return sub_results, ins_results

    def cache_probs(self, sources: SourceInferenceData):
        """
        Runs inferences in the model from given sources. 
        It is supposed that the context of the targets and their one-hots have already been computed in the model.
        """
        self.eval()
        with torch.no_grad():
            sub_results, ins_results = self(sources)
            x_l, y_l, batch_size = sub_results.size()[:-1] # |x|+1, |y|+1, batch_size

            # dim = (|x|+2, |y|+2, B)
            self.__cachedProbs = {key:np.zeros((x_l+1, y_l+1, batch_size)) for key in ('del', 'end', 'sub', 'ins')}

            self.__cachedProbs['del'][:-1, :-1] = sub_results[:, :, :,
                                                    self.output_dim-1].cpu().numpy()  # usefull space = (|x|+1, |y|+1, B)
            self.__cachedProbs['end'][:-1, :-1] = ins_results[:, :, :,
                                                    self.output_dim-1].cpu().numpy()  # usefull space = (|x|+1, |y|+1, B)

            targetsOneHots = self.__cachedTargetsData.nextOneHots
            # q(y[j]| x, i, y[:j]) = < onehot(y[j]), q(.| x, i, y[:j]) >
            self.__cachedProbs['sub'][:-1, :-2] = torch.nan_to_num(torch.logsumexp(
                sub_results[:, :-1, :, :-1] * targetsOneHots, dim=3), nan=0.).cpu().numpy()  # usefull space = (|x|+1, |y|, B)
            self.__cachedProbs['ins'][:-1, :-2] = torch.nan_to_num(torch.logsumexp(
                ins_results[:, :-1, :, :-1] * targetsOneHots, dim=3), nan=0.).cpu().numpy()  # usefull space = (|x|+1, |y|, B)

    def ins(self, i: int, j: int):
        return self.__cachedProbs['ins'][i, j]

    def sub(self, i: int, j: int):
        return self.__cachedProbs['sub'][i, j]

    def end(self, i: int, j: int):
        return self.__cachedProbs['end'][i, j]

    def dlt(self, i: int, j: int):
        return self.__cachedProbs['del'][i, j]
    
    #----------Training----------------

    def __computeMaskedProbDistribs(self, sub_probs:Tensor, ins_probs:Tensor, targetOneHotVectors:Tensor)->Tensor:
        """
        Returns a tensor of shape (|x|+1, |y|+1, b, 2*(|Σ|+1)) with a mask applied on each probability in the distribution which are not defined in the target probabilities cache. This is done according to the one-hots of the characters of interest in the target batch.

        Arguments:
            - sub_probs (Tensor): the logits or the targets for q_sub probs, with at the end the deletion prob
                dim = (|x|+1, |y|+1, b, |Σ|+1) or (|x|+1, |y|+1, b, 2) 
            - ins_probs (Tensor): the logits or the targets for q_ins probs, with at the end the insertion prob
                dim = (|x|+1, |y|+1, b, |Σ|+1) or (|x|+1, |y|+1, b, 2)
            - targetsOneHotVectors : dim = (1, |y|, b, |Σ|), the one hot vectors of the IPA characters in the modern forms mini batch.
            It is assumed that the padding mask has already been applied on the probs in argument.
        """
        b, V = targetOneHotVectors.shape[-2:]
        #TODO: computing one hots from targets mini batch
        nextOneHots = torch.cat((targetOneHotVectors, torch.zeros((1,1,b,V), device=device)), dim=1)
        masked_sub_probs = torch.nan_to_num(nextOneHots * sub_probs[:,:,:,:-1], nan=0.)
        masked_ins_probs = torch.nan_to_num(nextOneHots * ins_probs[:,:,:,:-1], nan=0.)
        deletionProbs = sub_probs[:,:,:,-1:]
        endingProbs = ins_probs[:,:,:,-1:]
        return torch.cat((masked_sub_probs, deletionProbs, masked_ins_probs, endingProbs), dim=-1)
    
    def get_targets(self, prob_targets_cache:ProbCache, samplesLengthData:tuple[Tensor, int], targetsLengthData:tuple[Tensor, int], targetOneHotVectors:Tensor):
        """
        This method computes the targets probs tensor with a good format and an applied padding mask, which is computed thanks to the samples' lengths IntTensor.

        ## targets distrib tensor shape = (|x|+1, |y|+1, b, 2*(|Σ|+1))

        ## Arguments:
            * prob_targets_cache (ProbCache) : the targets output probabilities computed from the backward dynamic program.
            * samplesLengthData (IntTensor, int): the CPU IntTensor/LongTensor with the samples' sequence lengths (with boundaries, so |x|+2) and the integer with the maximum one. It is expected data for the whole batch to be given.
            * targetsLengthData (IntTensor, int): the CPU IntTensor/LongTensor with the targets' sequence lengths (with boundaries, so |y|+2) and the integer with the maximum one. It is expected data for the whole batch to be given.
            * targetOneHotVectors (Tensor): dim = (1, |y|, b, |Σ|), the one hot vectors of the IPA characters in the modern forms mini batch.
            
            It is assumed that the padding mask has already been applied on the probs in argument.
        """
        with torch.no_grad():
            paddingMask = self.__computePaddingMask(samplesLengthData, targetsLengthData)
            
            prob_targets = self.__computeMaskedProbDistribs(
                torch.nan_to_num(paddingMask*torch.as_tensor(np.concatenate((
                                            np.expand_dims(prob_targets_cache.sub[:-1,:-1], -1), 
                                            np.expand_dims(prob_targets_cache.dlt[:-1,:-1], -1)
                                        ), axis=-1), device=device),nan=0.),
                torch.nan_to_num(paddingMask*torch.as_tensor(np.concatenate((
                                            np.expand_dims(prob_targets_cache.ins[:-1,:-1], -1),
                                            np.expand_dims(prob_targets_cache.end[:-1,:-1], -1)
                                        ), axis=-1), device=device),nan=0.),
                targetOneHotVectors
            )
        return prob_targets

    def get_logits(self, samples_:SourceInferenceData, targets_:TargetInferenceData, targetOneHotVectors:Tensor):
        """
        In training mode, computes the logits in a training step, from the specified samples and targets minibatch.
        logits tensor shape: (|x|+1 , |y|+1 , b, 2*(|Σ|+1)).
        """
        
        sub_outputs, ins_outputs = self(samples_, targets_)
        masked_outputs = self.__computeMaskedProbDistribs(sub_outputs, ins_outputs, targetOneHotVectors)
        
        return masked_outputs
