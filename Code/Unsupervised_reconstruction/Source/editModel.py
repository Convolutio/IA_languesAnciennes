#!/usr/local/bin/python3.9.10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from torch import Tensor
from Types.articleModels import *
from Types.models import InferenceData
from data.vocab import V_SIZE

device = "cuda" if torch.cuda.is_available() else "cpu"

BIG_NEG = -1e9

class EditModel(nn.Module):
    """
    # Neural Edit Model

    ### This class gathers the neural insertion and substitution models, specific to a branch between the\
          proto-language and a modern language.
    `q_ins` and `q_sub` are defined as followed:
        * Input: a sequence of N one-hot vectors of dimension |Σ|+2 representing an element\
        of (Σ ∪ {`'('`, `')'`})
        * Output: a tensor of dimension |Σ|+1 representing a probability distribution over\
        Σ ∪ {`'<del>'`} for `q_sub` or Σ ∪ {`'<end>'`} for `q_ins`
    #### Inference in the model
    With the `cache_probs` method.
    """
    def __init__(self, languages:ModernLanguages, hidden_dim: int, input_dim: int = V_SIZE+2, output_dim: int = V_SIZE+1):
        super(EditModel, self).__init__()
        self.languages = languages
        
        self.input_dim = input_dim # must equals |Σ|+2 for the ( and ) boundaries characters.
        self.hidden_dim = hidden_dim # abitrary embedding size
        self.output_dim = output_dim # must equals |Σ|+1 for the <end>/<del> characters
        
        self.encoder_prior = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.encoder_modern = nn.LSTM(input_dim, hidden_dim*2)

        self.sub_head = nn.Sequential(
            nn.Linear(hidden_dim*2, output_dim),
            nn.LogSoftmax()
            )
        self.ins_head = nn.Sequential(
            nn.Linear(hidden_dim*2, output_dim),
            nn.LogSoftmax()
            )
        
        self.__modernContext: Tensor
        self.__targetsLogOneHots: Tensor
        self.__cachedProbs: dict[Literal['ins', 'sub', 'end', 'del'], Tensor]

    def cache_target_context(self, targets_: InferenceData):
        """
        Optimisation method : computes once the usefull data for the targets for the probs computing over each Metropolis-Hastings iteration.
        """
        self.__modernContext, _ = pad_packed_sequence(self.encoder_modern(targets_[0])) # dim = (|y|+2, B, 2*hidden_dim)
        unpackedTargets, _ = pad_packed_sequence(targets_[0])
        self.__targetsLogOneHots = torch.log(unpackedTargets[:,1:,:,:-1]) #dim = (|x|+2, |y|+1, B, |Σ|+1) : the boundaries are not interesting values for y[j] (erased by the reduction)
    
    def cache_probs(self, sources: PackedSequence):
        """
        Runs inferences in the model from given sources. It is supposed that the context of the targets and their one-hots have already been computed in the model.
        """
        sourcePack = sources
        
        priorContext, _ = pad_packed_sequence(self.encoder_prior(sourcePack)) # dim = (|x|+2, B, 2*hidden_dim)
        modernContext = self.__modernContext
        priorN, modernN = priorContext.size()[0], modernContext.size()[0]
        ctx = priorContext.repeat(modernN, 1,1,1).transpose(0,1) + modernContext.repeat(priorN, 1,1,1) # dim = (|x|+2, |y|+2, B, 2*hidden_dim)
        
        sub_results = self.sub_head(ctx) # dim = (|x|+2, |y|+2, B, |Σ|+1) ; the ')' column is useless
        ins_results = self.ins_head(ctx) # dim = (|x|+2, |y|+2, B, |Σ|+1)
        
        #TODO: check if in the padded part the probs equal -infty
        self.__cachedProbs['del'] = sub_results[:,:,:,self.output_dim-1] #dim = (|x|+2, |y|+2, B)
        self.__cachedProbs['end'] = ins_results[:,:,:,self.output_dim-1] #dim = (|x|+2, |y|+2, B)
        
        targetsLogOneHots = self.__targetsLogOneHots
        # q(y[j]| x, i, y[:j]) = < onehot(y[j]), q(.| x, i, y[:j]) >
        self.__cachedProbs['sub'] = torch.logsumexp(sub_results[:,:-1,:,:] + targetsLogOneHots, dim=3) #dim = (|x|+2, |y|+1, B)
        self.__cachedProbs['ins'] = torch.logsumexp(ins_results[:,:-1,:,:] + targetsLogOneHots, dim=3) #dim = (|x|+2, |y|+1, B)
    
    
    def ins(self, i:int, j:int):
        return self.__cachedProbs['ins'][i, j, :]
    def sub(self, i:int, j:int):
        return self.__cachedProbs['sub'][i, j, :]
    def end(self, i:int, j:int):
        return self.__cachedProbs['end'][i, j, :]
    def dlt(self, i:int, j:int):
        return self.__cachedProbs['del'][i, j, :]
    

    
    def training(self, train_loader, test_loader, epochs=5, learning_rate=0.01):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
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

    