#!/usr/local/bin/python3.9.10
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from torch import Tensor
from models import ProbCache
from Types.articleModels import *
from Types.models import InferenceData
from Source.packingEmbedding import PackingEmbedding

device = "cuda" if torch.cuda.is_available() else "cpu"

BIG_NEG = -1e9


class CachedTargetsData():
    """
    Contains :
        * modernContext (Tensor) : the context embeddings computed from the targets
        * logarithmicOneHots (Tensor) : the logarithmics targets' one-hot vectors
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

    def __init__(self, targets_:InferenceData) -> None:
        """
        Optimisation method : computes once the usefull data for the targets at the EditModel's initialisation.
        The gradient of the targets input data is set in the method to be tracked.
        """
        self.maxSequenceLength = targets_[2]+2
        
        targets, sequencesLengths = targets_[0], targets_[1]

        closing_boundary_index = torch.max(targets)
        opening_boundary_index = closing_boundary_index - 1
        voc_size = closing_boundary_index - 2
        targetsInput = targets.where(targets == closing_boundary_index, 0)[:-1]
        targetsInput.requires_grad_(True)
        self.targetsInputData = targetsInput, targets_[1] + 1
        
        oneHots = torch.where(torch.logical_or(targets == closing_boundary_index, targets == 0), 0, targets - 1)[1:-1].to(torch.int64)
        original_shape = oneHots.size() # (|y|, B)
        # dim = (1, |y|, B, |Σ|) : the boundaries and special tokens are not interesting values for y[j] (that is why they have been erased with the reduction)
        self.nextOneHots = pad_packed_sequence(pack_padded_sequence(nn.functional.one_hot(oneHots.flatten(), num_classes=voc_size).view(original_shape+(voc_size,)), sequencesLengths, False, False), False)[0]
        
        self.arePaddingElements = torch.arange(self.maxSequenceLength-1).unsqueeze(0) < (sequencesLengths-1).unsqueeze(1)


class EditModel(nn.Module):
    """
    # Neural Edit Model

    ### This class gathers the neural insertion and substitution models, specific to a branch between the\
          proto-language and a modern language.
    `q_ins` and `q_sub` are defined as followed:
        * Reconstruction input (samples): a batch of packed sequences of one-hot indexes from 0 to |Σ|+2 representing phonetic or special tokens\
        in Σ ∪ {`'('`, `')'`}
        * Targets input (cognates): a batch of packed sequences of one-hot indexes from 0 to |Σ|+1 representing phonetic or special tokens in Σ ∪ {`'('`, `')'`}. The processing of the closing boundary at the end of the sequence will be avoided by the model thanks to the unidirectionnality of the context. 
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
            shared_embedding_layer,
            nn.LSTM(lstm_input_dim, lstm_hidden_dim//2, bidirectional=True).to(device)
        )
        self.encoder_modern = nn.Sequential(
            shared_embedding_layer,
            nn.LSTM(lstm_input_dim, lstm_hidden_dim).to(device)
        )

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
    
    #----------Inference---------------

    def update_cachedTargetContext(self, inferenceMode:bool = True):
        """
        Call this method at each EM iteration or at each training epoch to update the target's predicted context.
        """
        cache = self.__cachedTargetsData
        targets = cache.targetsInputData
        # TODO: make this code prettier
        if inferenceMode:
            with torch.no_grad():
                m = pad_packed_sequence(self.encoder_modern(targets[0], targets[1], batch_first=False)[0])[0] # dim = (|y|+1, B, 2*hidden_dim)
                cache.modernContext = m.movedim((1,2), (0,1)).unsqueeze(-2)
        else:
            m = pad_packed_sequence(self.encoder_modern(targets[0], targets[1], batch_first=False)[0])[0] # dim = (|y|+1, B, 2*hidden_dim)
            cache.modernContext = m.movedim((1,2), (0,1)).unsqueeze(-2)

    

    def __computeMask(self, sourcesLengths:Tensor, maxSourceLength:int):
        """
        Computes a mask for the padded elements from the reconstructions x and the targets y.
        
        Mask dim = (|x|+1, |y|+1, B, 1)

        Apply the mask with broadcasting as below
        >>> mask = self.__computeMask(sourcesLengths, maxSourceLength)
        >>> masked_ctx = mask*ctx

        Arguments:
            - sourcesLengths: a cpu tensor containing the lengths of the samples with the boundaries (|x|+2)
            - maxSourceLength: the value of the maximum sequence length (including the boundaries).
        """
        A = torch.arange(maxSourceLength-1).unsqueeze(0) < sourcesLengths.unsqueeze(1)-1 # dim = (B, |x|+1)
        B = self.__cachedTargetsData.arePaddingElements # dim = (B, |y|+1)
        return torch.logical_and(A.unsqueeze(-1), B.unsqueeze(-2)).to(device).movedim((1, 2), (0, 1)).unsqueeze(-1) # dim = (|x|+1, |y|+1, B, 1)
    
    
    def forward(self, sources_:InferenceData, inferenceMode:bool = True):
        """
        Returns a tuple of tensors representing respectively log q_sub(. |x,.,y[:.]) and log q_ins(.|x,.,y[:.]) , for each (x,y) sample-target couple in the batch.

        Tensors dimension = (|x|+1, |y|+1, B, |Σ|+1)
        
        The value for padded sequence elements is 0, which infers an undefined and neutral probability distribution for these elements. Moreover, if log(logit) = log(target) = 0, so the contribution of padded tokens in the chosen loss is neutralized.
        
        Arguments:
            - sources_ = (sources, lengths, maxSequenceLength), with sources a ByteTensor of dim = (|x|+2, B)
            - inferenceMode (boolean, True by default): if False, the context of the target cognates is updated with a gradient tracking.
        """
        if not inferenceMode:
            self.update_cachedTargetContext(False)

        priorContext, sourcesLengths = pad_packed_sequence(self.encoder_prior(
            sources_[0], sources_[1], False)[0])
        priorMaxSequenceLength = priorContext.size()[0] # |x|+2
        priorContext = priorContext[:-1] # dim = (|x|+1, B, 2*hidden_dim)
        ctx = (priorContext.movedim((1,2), (0,1)).unsqueeze(-1) + self.__cachedTargetsData.modernContext)\
            .movedim((2,3), (0,1)) # dim = (|x|+1, |y|+1, B, 2*hidden_dim)
        
        mask = self.__computeMask(sourcesLengths, priorMaxSequenceLength)
        masked_ctx = mask*ctx

        sub_results = self.sub_head(masked_ctx)*mask  # dim = (|x|+1, |y|+1, B, |Σ|+1)
        ins_results = self.ins_head(masked_ctx)*mask  # dim = (|x|+1, |y|+1, B, |Σ|+1)
        
        return sub_results, ins_results

    def cache_probs(self, sources: InferenceData):
        """
        Runs inferences in the model from given sources. It is supposed that the context of the targets and their one-hots have already been computed in the model.
        
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

    def __computeMaskedProbDistribs(self, sub_probs:Tensor, ins_probs:Tensor)->Tensor:
        """
        Returns a tensor of shape (|x|+1, |y|+1, B, 2*(|Σ|+1)) with a mask applied on each probability in the distribution which are not defined in the target probabilities cache. This is done according to the one-hots of the characters of interest in the target batch.

        Arguments:
            - sub_probs (Tensor): the logits or the targets for q_sub probs, with at the end the deletion prob
                dim = (|x|+1, |y|+1, B, |Σ|+1) or (|x|+1, |y|+1, B, 2) 
            - ins_probs (Tensor): the logits or the targets for q_ins probs, with at the end the insertion prob
                dim = (|x|+1, |y|+1, B, |Σ|+1) or (|x|+1, |y|+1, B, 2)
            - flatten (boolean): if the tensor has
            It is assumed that the padding mask has already been applied on the probs in argument.
        """
        B, V = self.__cachedTargetsData.nextOneHots.shape[-2:]
        nextOneHots = torch.cat((self.__cachedTargetsData.nextOneHots, torch.zeros((1,1,B,V), device=device)), dim=1)
        masked_sub_probs = torch.nan_to_num(nextOneHots * sub_probs[:,:,:,:-1], nan=0.)
        masked_ins_probs = torch.nan_to_num(nextOneHots * ins_probs[:,:,:,:-1], nan=0.)
        deletionProbs = sub_probs[:,:,:,-1:]
        endingProbs = ins_probs[:,:,:,-1:]
        return torch.cat((masked_sub_probs, deletionProbs, masked_ins_probs, endingProbs), dim=-1)

    def renderTargetAndLogitsBeforeLossComputation(self, samples_:InferenceData, prob_targets_cache:ProbCache):
        """
        Debugging method. Render respectively the target probs tensor and the logits tensor with the needed masking (padding and distribution masking) before the loss computation.
        tensor shape = (|x|+1, |y|+1, B, 2*(|Σ|+1))

        The arguments are the same than with the `training` method.
        """
        with torch.no_grad():
            samples = samples_[0]
            paddingMask = self.__computeMask(samples_[1]+2, samples_[2]+2)
            
            prob_targets = self.__computeMaskedProbDistribs(
                torch.nan_to_num(paddingMask*torch.as_tensor(np.concatenate((
                                            np.expand_dims(prob_targets_cache.sub[:-1,:-1], -1), 
                                            np.expand_dims(prob_targets_cache.dlt[:-1,:-1], -1)
                                        ), axis=-1), device=device),nan=0.),
                torch.nan_to_num(paddingMask*torch.as_tensor(np.concatenate((
                                            np.expand_dims(prob_targets_cache.ins[:-1,:-1], -1),
                                            np.expand_dims(prob_targets_cache.end[:-1,:-1], -1)
                                        ), axis=-1), device=device),nan=0.)
            )
            sub_outputs, ins_outputs = self(samples, False)
            masked_outputs = self.__computeMaskedProbDistribs(sub_outputs, ins_outputs)
        return prob_targets, masked_outputs
    
    def get_targets(self, prob_targets_cache:ProbCache, samples_:InferenceData):
        """
        This method computes the targets probs tensor with a good format and an applied padding mask, which is computed thanks to the samples' lengths IntTensor.

        targets distrib tensor shape = (|x|+1 * |y|+1 * B, 2*(|Σ|+1))
        """
        with torch.no_grad():
            paddingMask = self.__computeMask(samples_[1]+2, samples_[2]+2)
            
            prob_targets = self.__computeMaskedProbDistribs(
                torch.nan_to_num(paddingMask*torch.as_tensor(np.concatenate((
                                            np.expand_dims(prob_targets_cache.sub[:-1,:-1], -1), 
                                            np.expand_dims(prob_targets_cache.dlt[:-1,:-1], -1)
                                        ), axis=-1), device=device),nan=0.),
                torch.nan_to_num(paddingMask*torch.as_tensor(np.concatenate((
                                            np.expand_dims(prob_targets_cache.ins[:-1,:-1], -1),
                                            np.expand_dims(prob_targets_cache.end[:-1,:-1], -1)
                                        ), axis=-1), device=device),nan=0.).flatten(end_dim=-2)
            )
        return prob_targets

    def get_logits(self, samples_:InferenceData):
        """
        In training mode, computes the logits in a training step, from the specified samples.
        logits tensor shape: (|x|+1 * |y|+1 * B, 2*(|Σ|+1)).
        As the sources are common over all the EditModels, it is assumed that the samples's gradient
        has already been externally set up to tracking mode.
        """
        samples = samples_[0]
        assert(self.training), "This method must be called in training mode."
        assert(samples.requires_grad), "The source input data must have its gradient in tracking mode."
        
        sub_outputs, ins_outputs = self(samples, False)
        masked_outputs = self.__computeMaskedProbDistribs(sub_outputs, ins_outputs).flatten(end_dim=-2)
        
        return masked_outputs
    

class EditModels(nn.Module):
    def __init__(self, cognatesSet:dict[ModernLanguages, InferenceData], voc_size:int, lstm_input_dim:int, lstm_hidden_dim:int):
        super().__init__()
        self.__languages: tuple[ModernLanguages, ...] = tuple(cognatesSet.keys())
        self.shared_embedding_layer = PackingEmbedding(voc_size+3, lstm_input_dim, padding_idx=0)
        self.__editModels =  nn.ModuleDict[ModernLanguages, EditModel]({ # type: ignore
                lang:EditModel(
                    targetInferenceData,
                    lang,
                    self.shared_embedding_layer,
                    voc_size,
                    lstm_hidden_dim
                ) for (lang, targetInferenceData) in cognatesSet.items()
            })
    
    @property
    def models(self):
        return self.__editModels
    
    @property
    def languages(self):
        return self.__languages
    
    
    def train_models(self, samples_:InferenceData, prob_targets_cache:dict[ModernLanguages, ProbCache], epochs=5, learning_rate=0.01):
        samples_[0].requires_grad_(True)
        #TODO: put each income language with language for mini-batching
        targets_load = torch.cat([self.models[lang].get_targets(prob_targets_cache[lang], samples_) for lang in self.__languages], dim=0) # dim = (B_train, 2*(|Σ|+1))
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #TODO: experimenting smoothing values for this loss function.
        loss_function = nn.KLDivLoss(reduction="batchmean", log_target=True) # neutral if log(target) = log(logit) = 0
        
        self.train()

        for _ in range(epochs):
            optimizer.zero_grad()
            logits_load = torch.cat([self.models[lang].get_logits(samples_) for lang in self.__languages], dim=0) # dim = (B_train, 2*(|Σ|+1))
            
            loss = loss_function(logits_load, targets_load)
            loss.backward()
            optimizer.step()