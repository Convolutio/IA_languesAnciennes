from typing import Union
from math import log
from itertools import permutations

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, sgd

from Types.models import InferenceData

INFTY_NEG = -1e9
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PriorLM:
    def __init__(self, vocabTensors: list[Tensor]):     # list[Tensor], in future maybe one big Tensor 
        self.vocabTensors: list[Tensor]
        self.charDistrib: Tensor

    def train(self, data: Union[str, list[str]]) -> None:
        pass

    def evaluation(self) -> float:
        pass

    def inference(self, reconstructions: InferenceData) -> Tensor:
        pass


class NGramLM(PriorLM):
    def __init__(self, vocabTensors: list[Tensor], n: int):
        super().__init__(vocabTensors=vocabTensors)
        self.n: int = n 
        # self.ngramTensors: list[Tensor] = [torch.stack(p) for p in permutations(self.vocabTensors, self.n)]
        
        # log distribution
        self.distrib: dict[int, float] = {i: INFTY_NEG for i in range(len(self.ngramTensors))}   # Rework as Tensor 

    @staticmethod
    def countSubtensorOccurrences(larger_tensor: Tensor, sub_tensor: Tensor) -> int:
        """
        Count the number of sub-tensors that can exist in a large tensor.
        
        *This function was designed to count n-grams in a large tensor of size (L x B x V),
        each row of which corresponds to a character in a word in a column,
        and the depth of which corresponds to the one-hot tensor associated with that character.*

        >>> bigTensor = torch.tensor([[[0,0,1], [0,1,0], [0,0,1]],
                                        [[0,0,1], [0,1,0], [0,0,0]]])
        >>> littleTensor = torch.tensor([[0,0,1], [0,1,0]])
        >>> countSubtensorOccurrences(bigTensor, littleTensor)
        2
        """

        larger_reshaped = larger_tensor.unfold(1, sub_tensor.size()[0], 1).unfold(2, sub_tensor.size()[1], 1)   # Forms the n-grams.

        t = torch.all(larger_reshaped == sub_tensor, dim=2)
        mask = torch.all(t, dim=3)
        t_masked = t[mask]  # To keep only full True one hot character tensor.

        occurrences = torch.sum(t_masked).item()//(sub_tensor.size()[0]*sub_tensor.size()[1])

        return occurrences
    
    def train(self, data: Union[str, list[str]]):
        # Tokenize (call api tokenizer -> to keep only word in the voc from the data)
        # Vectorize (call make one hot)
        # Create a batch SAME AS RECONSTRUCTIONS in inference
        batch = torch.tensor()

        for i, t in enumerate(self.ngramTensors):
            self.distrib[i] += log(self.countSubtensorOccurrences(batch, t)) - log(self.countSubtensorOccurrences(batch, t[:-1])) #Rework as Tensor

    def evaluation(self):
        # Returns : perplexity of the model.
        pass

    def padDataWithN(self, reconstructions:InferenceData):
        """
        Returns the reconstructions in one-hot vectors tensor format, with ensuring that sequences with a length smaller than n are padded with the ')' token.
        Arguments :
            - reconstructions (InferenceData) : a tuple whose the first term is a tensor of (L, B, V) shape.
        """
        data, lengths = nn.utils.rnn.pad_packed_sequence(reconstructions[0], padding_value=1.0) # ngrams with empty characters have a neutral probability of 1.
        maxSequenceLength, batch_size, V = data.shape
        if maxSequenceLength < self.n:
            data = torch.cat((data, torch.zeros(self.n - maxSequenceLength, batch_size, V)), dim=0).to(device)
            maxSequenceLength = self.n
        a = torch.arange(maxSequenceLength).unsqueeze(0).expand(batch_size, -1)
        condition = torch.logical_and(lengths.unsqueeze(1).expand(-1, maxSequenceLength) <= a, a < self.n).T.unsqueeze(-1).expand(-1,-1,V) # size = (L, B, V)
        pad = torch.zeros((maxSequenceLength, batch_size, V), dtype=torch.float32).to(device)
        pad[:,:,V-1] = 1
        data = torch.where(condition, pad, data)
        return torch.log(data)

    def inference(self, reconstructions: InferenceData) -> Tensor:
        data = self.padDataWithN(reconstructions)
        maxSequenceLength, batch_size, V = data.size() 
        probs: Tensor = torch.zeros(batch_size, dtype=torch.float64) # begin with the neutral prob 1

        # For each wordord by th in the batch, get the probability of the we sum (cause we
        # are in log) of all conditionnal probability of each character composed the word. 
        for l in range(maxSequenceLength - self.n):
            nGram = data[l:l+self.n]
            nGramProb = nGramLogProbsTensor.unsqueeze(0).expand(batch_size, *(-1,)*self.n)
            for k in range(self.n):
                selected_nGram = nGram[k]
                for _ in range(self.n-k-1):
                    selected_nGram = selected_nGram.unsqueeze(-1)
                print(selected_nGram.size())
                selected_nGram = selected_nGram.expand(-1, -1, *(V,)*(self.n-k-1))
                nGramProb = torch.logsumexp(selected_nGram + nGramProb, dim=1)
            probs += torch.min(nGramProb, torch.zeros(batch_size)) # to neutralize the probability of ngrams with empty characters.
        return probs


class RNNLM(nn.Module, PriorLM):
    """
    Classic network predicting the next character of a string.

    Params:
        vocab_size (int) : The number of character in the vocabulary.
        embedding_size : Dimension of the character embedding vectors.
        hidden_size: Size of the LSTM hidden state.
        num_layers: Number of the layers of the LSTM.
        dropout_rate: Probability to drop out a neuron.    
    """

    def __init__(self, vocabTensors: list[Tensor], vocab_size: int, embedding_size: int, hidden_size: int, num_layers: int, dropout_rate: float):
        super(RNNLM, self).__init__(vocabTensors=vocabTensors)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embedding_size)

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=self.num_layers,
                            dropout=self.dropout_rate)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, prev_state):
        embedded = self.embedding(x)
        output, state = self.lstm(embedded, prev_state)
        output = self.dropout(output)
        output = self.fc(output)
        return output, state

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

    def train(self, data: Union[str, list[str]], epochs: int):
        criter = nn.CrossEntropyLoss()
        optim = optim.Adam(self.parameters(), lr=0.001)
        # Tokenize
        # Vectorize
        # Dataloader
        for epoch in tqdm(range(epoch), desc=f'Training model for epoch {epoch}/{epochs}', total=epochs):
            for batch_idx, (data, trgt) in enumerate(train_data):
                scores = self(data)
                loss = criter(scores, trgt)
                optim.zero_grad()
                loss.backward()
                optim.step()
                print(f'epoch: {epoch} step: {batch_idx + 1}/{len(train_data)} loss: {loss}')

    def evaluation(self):
        # Returns : perplexity of the model.
        pass

    def inference(self, reconstructions: InferenceData) -> Tensor:
        pass