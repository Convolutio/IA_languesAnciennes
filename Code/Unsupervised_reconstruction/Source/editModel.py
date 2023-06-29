#!/usr/local/bin/python3.9.10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from torch import Tensor
from Types.articleModels import *
from Types.models import InferenceData
from typing import Optional
from data.vocab import V_SIZE

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
    Context of y, ready to be summed with the context of y.

    dim = (B, 2*hidden_dim, 1, |y|+2)
    """
    
    logarithmicOneHots: Tensor
    sequencesLengths: Tensor
    maxSequenceLength: int
    
    arePaddingElements: Tensor
    """
    dim = (B, |y|+2)
    """

    def __init__(self, targets_:InferenceData) -> None:
        """
        Optimisation method : computes once the usefull data for the targets for the probs computing over each Metropolis-Hastings iteration.
        """
        self.maxSequenceLength = targets_[2]
        
        unpackedTargets, self.sequencesLengths = pad_packed_sequence(targets_[0])
        # dim = (|y|+1, B, |Σ|+1) : the boundaries are not interesting values for y[j] (that is why they have been erased with the reduction)
        self.logarithmicOneHots = torch.log(unpackedTargets[1:, :, :-1])

        self.arePaddingElements = torch.arange(self.maxSequenceLength, device=device).unsqueeze(0) < torch.as_tensor(targets_[1], dtype=torch.int8, device=device).unsqueeze(1)


class EditModel(nn.Module):
    """
    # Neural Edit Model

    ### This class gathers the neural insertion and substitution models, specific to a branch between the\
          proto-language and a modern language.
    `q_ins` and `q_sub` are defined as followed:
        * Input: a batch of packed sequences of one-hot vectors of dimension |Σ|+2 representing phonetic or special tokens\
        in (Σ ∪ {`'('`, `')'`})
        * Output: a tensor of dimension |Σ|+1 representing a probability distribution over\
        Σ ∪ {`'<del>'`} for `q_sub` or Σ ∪ {`'<end>'`} for `q_ins`

    ### Instructions for targets' data caching
        * At the beginning of the Monte-Carlo training, cache the initial targets' background data by adding an InferenceData object in __init__'s argument
        * At the beginning of each EM iteration, run the inference of the targets' context once with the `cache_target_context` method, so the context will be computed once for the sampling step and not at each MH sampling iteration. 
    
    ### Inference in the model
    With the `cache_probs` method.

    #### About the padding
    In the cached probs, some probabilities are undefined and has neutral logarithmic probability equaling 0.
    """

    def __init__(self, targets_:InferenceData, languages: ModernLanguages, hidden_dim: int, input_dim: int = V_SIZE+2, output_dim: int = V_SIZE+1):
        super(EditModel, self).__init__()
        self.languages = languages

        # must equals |Σ|+2 for the ( and ) boundaries characters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # abitrary embedding size
        self.output_dim = output_dim  # must equals |Σ|+1 for the <end>/<del> characters

        self.encoder_prior = nn.LSTM(input_dim, hidden_dim, bidirectional=True).to(device)
        self.encoder_modern = nn.LSTM(input_dim, hidden_dim*2).to(device)

        self.sub_head = nn.Sequential(
            nn.Linear(hidden_dim*2, output_dim),
            nn.LogSoftmax(dim=-1)
        ).to(device)
        self.ins_head = nn.Sequential(
            nn.Linear(hidden_dim*2, output_dim),
            nn.LogSoftmax(dim=-1)
        ).to(device)

        self.__cachedTargetsData = CachedTargetsData(targets_)
        
        self.__cachedProbs: dict[Literal['ins',
                                         'sub', 'end', 'del'], Tensor] = {}
        
    def cache_target_context(self, targets:PackedSequence, inferenceMode:bool = True):
        """
        Call this method at each EM iteration or at each training epoch to update the target's predicted context.

        Arguments:
            targets (PackedSequence)
            inferenceMode (bool) : set to True if the context is computed in an inference step, so the gradient is not tracked.
        """
        cache = self.__cachedTargetsData
        # TODO: make this code prettier
        if inferenceMode:
            with torch.no_grad():
                m, _ = pad_packed_sequence(
                    self.encoder_modern(targets))  # dim = (|y|+2, B, 2*hidden_dim)
                cache.modernContext = m.movedim((1,2), (0,1)).unsqueeze(-2)
        else:
            m, _ = pad_packed_sequence(
                self.encoder_modern(targets))  # dim = (|y|+2, B, 2*hidden_dim)
            cache.modernContext = m.movedim((1,2), (0,1)).unsqueeze(-2)

    

    def __computeMask(self, sourcesLengths:Tensor, maxSourceLength:int):
        """
        Computes a mask for the padded elements from the reconstructions x and the targets y.
        
        Mask dim = (|x|+2, |y|+2, B, 1)

        Apply the mask with broadcasting as below
        >>> mask = self.__computeMask(sourcesLengths, maxSourceLength)
        >>> masked_ctx = mask*ctx
        """
        A = torch.arange(maxSourceLength, device=device).unsqueeze(0) < sourcesLengths.unsqueeze(1) # dim = (B, |x|+2)
        B = self.__cachedTargetsData.arePaddingElements # dim = (B, |y|+2)
        return torch.logical_and(A.unsqueeze(-1), B.unsqueeze(-2)).movedim((1, 2), (0, 1)).unsqueeze(-1) # dim = (|x|+2, |y|+2, B, 1)
    
    def forward(self, sources:PackedSequence, targets:Optional[PackedSequence] = None):
        """
        Returns a tuple of tensors representing respectively log q_sub(. |x,.,y[:.]) and log q_ins(.|x,.,y[:.]) , for each (x,y) sample-target couple in the batch.

        Tensors dimension = (|x|+2, |y|+2, B, |Σ|+1)
        
        The value for padded sequence elements is 0, which infers an undefined and neutral probability distribution for these elements.
        
        Arguments:
            - sources (PackedSequence): the packed sequence with the one hot vectors representing each batch's reconstruction
            - targets (Optional[PackedSequence]): if given, the cache of the context of the target will be updated with gradient tracking (we assume we are in a training stage). 
        """
        sourcePack = sources
        if targets is not None:
            self.cache_target_context(targets, False)

        priorContext, sourcesLengths = pad_packed_sequence(self.encoder_prior(
            sourcePack)[0]) # priorContext dim = (|x|+2, B, 2*hidden_dim)
        priorMaxSequenceLength = priorContext.size()[0]
        ctx = (priorContext.movedim((1,2), (0,1)).unsqueeze(-1) + self.__cachedTargetsData.modernContext)\
            .movedim((2,3), (0,1)) # dim = (|x|+2, |y|+2, B, 2*hidden_dim)
        # ctx = priorContext.repeat(modernMaxSequenceLength, 1, 1, 1).transpose(
        #     0, 1) + modernContext.repeat(priorMaxSequenceLength, 1, 1, 1)  # dim = (|x|+2, |y|+2, B, 2*hidden_dim)

        mask = self.__computeMask(sourcesLengths, priorMaxSequenceLength)
        masked_ctx = mask*ctx

        sub_results = self.sub_head(masked_ctx)  # dim = (|x|+2, |y|+2, B, |Σ|+1)
        ins_results = self.ins_head(masked_ctx)  # dim = (|x|+2, |y|+2, B, |Σ|+1)
        return sub_results, ins_results
    
    # TODO: be sure this old method is useless
    # def __computePaddingMatrix(self, sourcesLengths: Tensor, maxSourceLength: int):
    #     """
    #     Computes a bool matrix where True represents an output value of interest and False a padding value (0).
    #     dim = (|x|+2, |y|+2, B, |Σ|+1)

    #     Arguments:
    #         sourcesLengths (IntTensor, dim = (B)) : contains the length of each reconstruction sequence
    #         maxSourceLength (int)
    #     """
    #     batch_size = len(sourcesLengths)
    #     outputDim = self.output_dim
    #     maxTargetLength = self.__cachedTargetsData.maxSequenceLength
    #     A = torch.arange(maxSourceLength).repeat(outputDim, batch_size, 1).transpose(
    #         0, 2) < sourcesLengths.repeat(maxSourceLength, outputDim, 1).transpose(1, 2)  # dim = (|x|+2, B, |Σ|+1)
    #     B = self.__cachedTargetsData.arePaddingElements
    #     # dim = (|x|+2, |y|+2, B, |Σ|+1)
    #     return torch.logical_and(A.repeat(maxTargetLength, 1, 1, 1).transpose(0, 1), B.repeat(maxSourceLength, 1, 1, 1))

    def cache_probs(self, sources: PackedSequence):
        """
        Runs inferences in the model from given sources. It is supposed that the context of the targets and their one-hots have already been computed in the model.
        
        """
        with torch.no_grad():
            sub_results, ins_results = self(sources)

            self.__cachedProbs['del'] = sub_results[:, :, :,
                                                    self.output_dim-1].numpy()  # dim = (|x|+2, |y|+2, B)
            self.__cachedProbs['end'] = ins_results[:, :, :,
                                                    self.output_dim-1].numpy()  # dim = (|x|+2, |y|+2, B)

            targetsLogOneHots = self.__cachedTargetsData.logarithmicOneHots.repeat(
                sub_results.shape[0], 1, 1, 1)
            # q(y[j]| x, i, y[:j]) = < onehot(y[j]), q(.| x, i, y[:j]) >
            self.__cachedProbs['sub'] = torch.logsumexp(
                sub_results[:, :-1, :, :-1] + targetsLogOneHots, dim=3).numpy()  # dim = (|x|+2, |y|+1, B)
            self.__cachedProbs['ins'] = torch.logsumexp(
                ins_results[:, :-1, :, :-1] + targetsLogOneHots, dim=3).numpy()  # dim = (|x|+2, |y|+1, B)

    def ins(self, i: int, j: int):
        return self.__cachedProbs['ins'][i, j, :]

    def sub(self, i: int, j: int):
        return self.__cachedProbs['sub'][i, j, :]

    def end(self, i: int, j: int):
        return self.__cachedProbs['end'][i, j, :]

    def dlt(self, i: int, j: int):
        return self.__cachedProbs['del'][i, j, :]

    def training(self, samples_:InferenceData, targets:PackedSequence, prob_targets:Tensor, test_loader, epochs=5, learning_rate=0.01):
        samples = samples_[0]
        mask = self.__computeMask(torch.as_tensor(samples_[1], dtype=torch.int8, device=device), samples_[2])
        masked_prob_targets = prob_targets*mask

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            sub_outputs, ins_outputs = self(samples, targets)

            #TODO: compute the loss and backward

        # for epoch in range(epochs):
        #     running_loss = 0.0
        #     for inputs in train_loader:
        #         optimizer.zero_grad()
        #         outputs = self(inputs)
        #         loss = outputs.loss
        #         loss.backward()
        #         optimizer.step()
        #         running_loss += loss.item()

        #     print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
        #     self.evaluation(test_loader)

    def evaluation(self, test_loader):
        """
        Evaluate the performance of our model by computing the average edit distance between its outputs
        and gold Latin reconstructions.
        """
        correct = 0
        total = 0
        # with torch.no_grad():
        #     for data in test_loader:
        #         outputs = self(data)
        #         _, predicted = torch.max(outputs.data, 1)
        #         # edit distance

        # print(f'Test Accuracy: {100 * correct / total}%')
