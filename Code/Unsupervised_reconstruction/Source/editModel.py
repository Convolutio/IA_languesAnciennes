#!/usr/local/bin/python3.9.10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import mps
from Types.articleModels import *
from data.vocab import V_SIZE

device = "cuda" if torch.cuda.is_available() else "cpu"

BIG_NEG = -1e9

class EditModel(nn.Module):
    """
    This class gathers the edit models and edit function specific to a branch between the\
          proto-language and a modern language.
    q_ins and q_sub are defined as followed:
        Input: a sequence of N one-hot tensors of dimension |Σ|+2 representing an element\
        of (Σ ∪ {'(', ')'})
        Output: a tensor of dimension |Σ|+1 representing a probability distribution over\
        Σ ∪ {'<del>'} for q_sub or Σ ∪ {'<end>'} for q_ins
    """
    def __init__(self, languages:Languages, input_dim: int, hidden_dim: int, output_dim: int):
        super(EditModel, self).__init__()
        self.languages = languages

        self.input_dim = input_dim # must equals |Σ|+2 for the ( and ) boundaries characters.
        self.hidden_dim = hidden_dim # abitrary embedding size
        self.output_dim = output_dim # must equals |Σ|+1 for the <end>/<del> characters
        
        self.encoder_prior = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        #les dimensions des couches cachées sont multipliées par 2 pour être égale à
        #celle de la couche cachée de l'encodeur bidirectionnel (qui subit une concaténation)
        self.encoder_modern = nn.LSTM(input_dim, hidden_dim*2)

        self.sub_head = nn.Sequential(
            nn.Linear(hidden_dim*2, output_dim),
            nn.LogSoftmax() #we want the log of the probabilities
            )
        self.ins_head = nn.Sequential(
            nn.Linear(hidden_dim*2, output_dim),
            nn.LogSoftmax()
            )

    def __encode_context(self, x:Form, i:int, y0:Form):
        encoder_prior__output, _ = self.encoder_prior(x) # cette inférence s'exécute plusieurs fois inutilement !!
        h_xi = encoder_prior__output[i, :]
        encoder_modern__output, _ = self.encoder_modern(y0)
        g_y0 = encoder_modern__output[-1, :]
        return h_xi + g_y0
    
    def cache_probs(self, sources:OneHotsSet, targets:OneHotsSet):
        """
        Arguments:
            sources (OneHotsSet): the set of vectorized proto-forms with boundaries
            targets (OneHotsSet): the set of their vectorized target cognates with boundaries\
            in the language of this instance of EditModel.
        Computes the inference for each proto-form--cognate pair. 
        """
        #TODO revoir l'inférence
        #TODO : s'assurer que les caractères vides renvoient des résultats contextuels nuls
        #TODO : aller chercher l
        _, batch_size, INPUT_VOC_SIZE = sources.shape
        assert(INPUT_VOC_SIZE==self.input_dim), "The input dimension of the model is wrongly defined."
        max_protoForm_length = sources.shape[0]-2 # without boundaries
        max_cognat_length = targets.shape[0]-2 # without boundaries
        cachedProbsNDArray_shape = (max_protoForm_length+2, max_cognat_length+2, batch_size)
        self.__cachedProbs = {op:torch.as_tensor(np.full(cachedProbsNDArray_shape, BIG_NEG))
                              for op in ('ins', 'sub', 'end', 'del')}
        for b in range(batch_size):
            x, y = torch.as_tensor(sources[:, b, :]), torch.as_tensor(targets[:, b, :])
            x_embeddings, y_embedding = self.encoder_prior(x), self.encoder_modern(y)
            for i in range(max_protoForm_length+2):
                for j in range(max_cognat_length+1):
                    if True in x[i, :] and True in y[j, :]:
                        context = x_embeddings[i, :] + y_embedding[j, :]
                        subProbs, insProbs = self.sub_head(context), self.ins_head(context)
                        if not bool(y[j+1, V_SIZE+1]): # The character to be inserted/used in substitution is not ')'
                            self.__cachedProbs["ins"][i,j,b] = torch.inner(insProbs, y[j+1, :-2])
                            self.__cachedProbs["sub"][i,j,b] = torch.inner(subProbs, y[j+1, :-2])
                        self.__cachedProbs["end"][i,j,b] = insProbs[V_SIZE]
                        self.__cachedProbs["del"][i,j,b] = subProbs[V_SIZE]
    
    def ins(self, i:int, j:int):
        return self.__cachedProbs['ins'][i, j, :]
    def sub(self, i:int, j:int):
        return self.__cachedProbs['sub'][i, j, :]
    def end(self, i:int, j:int):
        return self.__cachedProbs['end'][i, j, :]
    def dlt(self, i:int, j:int):
        return self.__cachedProbs['del'][i, j, :]
    
    def q_sub(self, x:Form, i:int, y0:Form):
        """
        Forwards inference in q_sub model.
        Returns a tensor (dimension : |Σ*|) representing a logarithmic
        probability distribution over Σ* that each token Σ*[j] is added
        to y0 in replacing the x[i] token.
        """
        context = self.__encode_context(x, i, y0)
        return self.sub_head(context)
    
    def q_ins(self, x:Form, i:int, y0:Form)->torch.Tensor:
        """
        Forwards inference in q_ins model.
        Returns a tensor (dimension : |Σ*|) representing a probability
        distribution over Σ* that each token Σ*[j] is inserted after y0.
        """
        context = self.__encode_context(x, i, y0)
        return self.sub_ins(context)
    
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

    def edit(self, x: Form) -> tuple[Form, list[Edition]]:
        """
        Args:
            x (Form): a string of IPA tokens representing the ancestral form
        
        Returns
            y : a string representing a modern form 
            delta: list of computed editions
        """
        y0: Form = "" # it is represented as y' in the figure 2 of the paper
        delta: list[Edition] = []
        for i in range(len(x)):
            omega:IPA_char = SIGMA_SUB[torch.argmax(self.q_sub(x, i, y0))]
            delta.append(
                (op.sub, omega, x, i, y0)
                )
            if omega!='<del>':
                canInsert:bool = True
                while canInsert:
                    y0 += omega
                    omega = SIGMA_INS[torch.argmax(self.q_ins(x, i, y0))]
                    delta.append(
                        (op.ins, omega, x, i, y0)
                    )
                    canInsert = omega!="<end>"
        return (y0, delta)