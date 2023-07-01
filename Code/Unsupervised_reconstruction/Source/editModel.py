#!/usr/local/bin/python3.9.10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from torch import Tensor
from models import ProbCache
from Types.articleModels import *
from Types.models import InferenceData
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
    Context of y, ready to be summed with the context of x.

    dim = (B, 2*hidden_dim, 1, |y|+1)
    """

    inputOneHots: PackedSequence
    """
    The input one-hot vectors of the target cognates, without their ) closing boundary.

    dim = (|y|+1, B, |Σ|+1)
    """
    
    logarithmicOneHots: Tensor
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
        Optimisation method : computes once the usefull data for the targets for the probs computing over each Metropolis-Hastings iteration.
        """
        self.maxSequenceLength = targets_[2]+2
        
        unpackedTargets, sequencesLengths = pad_packed_sequence(targets_[0])
        
        self.inputOneHots = pack_padded_sequence(unpackedTargets[:-1, :, :-1], sequencesLengths-1, enforce_sorted=False)
        
        # dim = (1, |y|, B, |Σ|) : the boundaries and special tokens are not interesting values for y[j] (that is why they have been erased with the reduction)
        self.logarithmicOneHots = torch.log(unpackedTargets[1:-1, :, :-2]).unsqueeze(0)

        self.arePaddingElements = torch.arange(self.maxSequenceLength-1).unsqueeze(0) < (sequencesLengths-1).unsqueeze(1)


class EditModel(nn.Module):
    """
    # Neural Edit Model

    ### This class gathers the neural insertion and substitution models, specific to a branch between the\
          proto-language and a modern language.
    `q_ins` and `q_sub` are defined as followed:
        * Reconstruction input (samples): a batch of packed sequences of one-hot vectors of dimension |Σ|+2 representing phonetic or special tokens\
        in Σ ∪ {`'('`, `')'`}
        * Targets input (cognates): a batch of packed sequences of one-hot vectors of dimension |Σ|+2 representing phonetic or special tokens in Σ ∪ {`'('`, `')'`}. The processing of the closing boundary at the end of the sequence will be avoided by the model thanks to the unidirectionnality of the context. 
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

    def __init__(self, targets_:InferenceData, languages: ModernLanguages, hidden_dim: int, recons_input_dim: int = V_SIZE+2, target_input_dim:int = V_SIZE+1, output_dim: int = V_SIZE+1):
        super(EditModel, self).__init__()
        self.languages = languages

        # must equals |Σ|+2 for the ( and ) boundaries characters.
        self.recons_input_dim = recons_input_dim
        # equal |Σ|+1 for just the ( opening boudary character.
        self.target_input_dim = target_input_dim
        self.hidden_dim = hidden_dim  # abitrary embedding size
        self.output_dim = output_dim  # must equals |Σ|+1 for the <end>/<del> characters

        self.encoder_prior = nn.LSTM(recons_input_dim, hidden_dim, bidirectional=True).to(device)
        self.encoder_modern = nn.LSTM(target_input_dim, hidden_dim*2).to(device)

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
                                         'sub', 'end', 'del'], np.ndarray] = {}
        
    def update_cachedTargetContext(self, inferenceMode:bool = True):
        """
        Call this method at each EM iteration or at each training epoch to update the target's predicted context.

        Arguments:
            targets (PackedSequence) : dim = (|y|+1, B, |Σ|+1)
            inferenceMode (bool) : set to True if the context is computed in an inference step, so the gradient is not tracked.
        """
        cache = self.__cachedTargetsData
        targets = cache.inputOneHots
        # TODO: make this code prettier
        if inferenceMode:
            with torch.no_grad():
                m = pad_packed_sequence(self.encoder_modern(targets)[0])[0] # dim = (|y|+1, B, 2*hidden_dim)
                cache.modernContext = m.movedim((1,2), (0,1)).unsqueeze(-2)
        else:
            m = pad_packed_sequence(self.encoder_modern(targets)[0])[0] # dim = (|y|+1, B, 2*hidden_dim)
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
            - maxSourceLength: the value of maximum length.
        """
        A = torch.arange(maxSourceLength-1).unsqueeze(0) < sourcesLengths.unsqueeze(1)-1 # dim = (B, |x|+1)
        B = self.__cachedTargetsData.arePaddingElements # dim = (B, |y|+1)
        return torch.logical_and(A.unsqueeze(-1), B.unsqueeze(-2)).to(device).movedim((1, 2), (0, 1)).unsqueeze(-1) # dim = (|x|+1, |y|+1, B, 1)
    
    def forward(self, sources:PackedSequence, inferenceMode:bool = True):
        """
        Returns a tuple of tensors representing respectively log q_sub(. |x,.,y[:.]) and log q_ins(.|x,.,y[:.]) , for each (x,y) sample-target couple in the batch.

        Tensors dimension = (|x|+1, |y|+1, B, |Σ|+1)
        
        The value for padded sequence elements is 0, which infers an undefined and neutral probability distribution for these elements. Moreover, if log(logit) = log(target) = 0, so the contribution of padded tokens in the chosen loss is neutralized.
        
        Arguments:
            - sources (PackedSequence): the packed sequence with the one hot vectors representing each batch's reconstruction
            dim = (|x|+2, B, |Σ|+2)
            - inferenceMode (boolean, True by default): if False, the context of the target cognates is updated with a gradient tracking.
        """
        sourcePack = sources
        if not inferenceMode:
            self.update_cachedTargetContext(False)

        priorContext, sourcesLengths = pad_packed_sequence(self.encoder_prior(
            sourcePack)[0])
        priorMaxSequenceLength = priorContext.size()[0] # |x|+2
        priorContext = priorContext[:-1] # dim = (|x|+1, B, 2*hidden_dim)
        ctx = (priorContext.movedim((1,2), (0,1)).unsqueeze(-1) + self.__cachedTargetsData.modernContext)\
            .movedim((2,3), (0,1)) # dim = (|x|+1, |y|+1, B, 2*hidden_dim)
        
        mask = self.__computeMask(sourcesLengths, priorMaxSequenceLength)
        masked_ctx = mask*ctx

        sub_results = self.sub_head(masked_ctx)*mask  # dim = (|x|+1, |y|+1, B, |Σ|+1)
        ins_results = self.ins_head(masked_ctx)*mask  # dim = (|x|+1, |y|+1, B, |Σ|+1)
        #TODO: mask outputs which don't have to be defined for substitution or insertion operation.
        return sub_results, ins_results

    def cache_probs(self, sources: PackedSequence):
        """
        Runs inferences in the model from given sources. It is supposed that the context of the targets and their one-hots have already been computed in the model.
        
        """
        with torch.no_grad():
            sub_results, ins_results = self(sources)
            x_l, y_l, batch_size = sub_results.size()[:-1] # |x|+1, |y|+1, batch_size

            # dim = (|x|+2, |y|+2, B)
            self.__cachedProbs = {key:np.zeros((x_l+1, y_l+1, batch_size)) for key in ('del', 'end', 'sub', 'ins')}

            self.__cachedProbs['del'][:-1, :-1] = sub_results[:, :, :,
                                                    self.output_dim-1].cpu().numpy()  # usefull space = (|x|+1, |y|+1, B)
            self.__cachedProbs['end'][:-1, :-1] = ins_results[:, :, :,
                                                    self.output_dim-1].cpu().numpy()  # usefull space = (|x|+1, |y|+1, B)

            targetsLogOneHots = self.__cachedTargetsData.logarithmicOneHots
            # q(y[j]| x, i, y[:j]) = < onehot(y[j]), q(.| x, i, y[:j]) >
            self.__cachedProbs['sub'][:-1, :-2] = torch.nan_to_num(torch.logsumexp(
                sub_results[:, :-1, :, :-1] + targetsLogOneHots, dim=3), neginf=0.).cpu().numpy()  # usefull space = (|x|+1, |y|, B)
            self.__cachedProbs['ins'][:-1, :-2] = torch.nan_to_num(torch.logsumexp(
                ins_results[:, :-1, :, :-1] + targetsLogOneHots, dim=3), neginf=0.).cpu().numpy()  # usefull space = (|x|+1, |y|, B)

    def ins(self, i: int, j: int):
        return self.__cachedProbs['ins'][i, j]

    def sub(self, i: int, j: int):
        return self.__cachedProbs['sub'][i, j]

    def end(self, i: int, j: int):
        return self.__cachedProbs['end'][i, j]

    def dlt(self, i: int, j: int):
        return self.__cachedProbs['del'][i, j]
    
    def __computeMaskedProbTargetProb(self, prob_targets:ProbCache):
        """
        dim = (|x|+2 * |y|+1 * B, 2*(|Σ|+1))
        The results that are not computed are masked with 0 value.
        """
        masked_target_sub = torch.nan_to_num(self.__cachedTargetsData.logarithmicOneHots + torch.as_tensor(prob_targets.sub, device=device)[:,:-1].unsqueeze(-1), neginf=0.)
        masked_target_ins = torch.nan_to_num(self.__cachedTargetsData.logarithmicOneHots + torch.as_tensor(prob_targets.ins, device=device)[:,:-1].unsqueeze(-1), neginf=0.)
        masked_target_sub = torch.cat((masked_target_sub, torch.as_tensor(prob_targets.dlt, device=device)), dim=-1)
        masked_target_ins = torch.cat((masked_target_ins, torch.as_tensor(prob_targets.end, device=device)), dim=-1)
        return torch.cat((masked_target_sub, masked_target_ins), dim=-1).flatten(end_dim=-2)
    
    def __computeMaskedProbDistribs(self, sub_probs:Tensor, ins_probs:Tensor)->Tensor:
        """
        Returns a tensor of shape (|x|+1 * |y|+1 * B, 2*(|Σ|+1)) with a mask applied on each probability in the distribution which are not defined in the target probabilities cache. This is done according to the one-hots of the characters of interest in the target batch.

        Arguments:
            - sub_probs (Tensor): the logits or the targets for q_sub probs, with at the end the deletion prob
                dim = (|x|+1, |y|+1, B, |Σ|+1) or (|x|+1, |y|+1, B, 2) 
            - ins_probs (Tensor): the logits or the targets for q_ins probs, with at the end the insertion prob
                dim = (|x|+1, |y|+1, B, |Σ|+1) or (|x|+1, |y|+1, B, 2)
        """
        B, V = self.__cachedTargetsData.logarithmicOneHots.shape[-2:]
        logOneHots = torch.cat((self.__cachedTargetsData.logarithmicOneHots, torch.zeros((1,1,B,V), device=device)), dim=1)
        masked_sub_probs = torch.nan_to_num(logOneHots + sub_probs[:,:,:,:-1], neginf=0.)
        deletionProbs = sub_probs[:,:,:,-1]
        masked_ins_probs = torch.nan_to_num(logOneHots + ins_probs[:,:,:,:-1], neginf=0.)
        endingProbs = ins_probs[:,:,:,-1]
        return torch.cat((masked_sub_probs, deletionProbs, masked_ins_probs, endingProbs), dim=-1).flatten(end_dim=-2)


    def training(self, samples_:InferenceData, targets:PackedSequence, prob_targets_cache:ProbCache, test_loader, epochs=5, learning_rate=0.01):
        samples = samples_[0]
        # mask = self.__computeMask(torch.as_tensor(samples_[1], dtype=torch.int8, device=device), samples_[2])
        # masked_prob_targets = prob_targets*mask
        prob_targets = self.__computeMaskedProbDistribs(
            torch.as_tensor(np.concatenate((np.expand_dims(prob_targets_cache.sub, -1), 
                                            np.expand_dims(prob_targets_cache.dlt, -1)), axis=-1), device=device),
            torch.as_tensor(np.concatenate((np.expand_dims(prob_targets_cache.ins, -1),
                                            np.expand_dims(prob_targets_cache.end, -1)), axis=-1), device=device)
        )

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss = nn.KLDivLoss(reduction="batchmean", log_target=True) # neutral if log(target) = log(logit) = 0

        for _ in range(epochs):
            optimizer.zero_grad()
            sub_outputs, ins_outputs = self(samples, False)
            masked_outputs = self.__computeMaskedProbDistribs(sub_outputs, ins_outputs)
            l = loss(masked_outputs, prob_targets)
            l.backward()

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
