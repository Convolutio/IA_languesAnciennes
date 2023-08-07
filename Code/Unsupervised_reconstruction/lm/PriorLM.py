from itertools import product

import torch
import torch.nn as nn
from torch import Tensor
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm.auto import tqdm

from Types.models import InferenceData
from Data.vocab import wordsToOneHots, computeInferenceData, vocabulary, PADDING_TOKEN
from Source.packingEmbedding import PackingEmbedding

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PriorLM:
    def __init__(self):
        pass

    def train(self, data: str) -> None: ...

    def evaluation(self) -> float: ...

    def inference(self, reconstructions: InferenceData) -> Tensor: ...


class NGramLM(PriorLM):
    def __init__(self, n: int, vocab: Vocab = vocabulary):
        self.n = n
        self.vocab = vocab
        self.vocabSize = len(vocab)

        self.nGramCount = torch.zeros(
            (self.vocabSize,) * self.n, device=device)
        self.nGramLogProbs = torch.log(
            torch.zeros((self.vocabSize,) * self.n, device=device))

    def prepare(self, data: str) -> tuple[Tensor, Tensor, Tensor]:
        """
        Prepare the data in a ngram batch (shape: (*, self.n)) by checking the validity \
        of the vocabulary and giving the count of each unique ngram in the batch (shape: (*)).
        The last tensor corresponds to a BoolTensor (shape: (*)) checking for each unique ngram if containing a 0.

        Args:
            data (str): the training data.

        Returns:
            tuple[Tensor, Tensor, Tensor]: one containing each unique ngram, \
            an other containing the counts for each ngram and the last one checking if a ngram contains a 0.
        """
        assert len(d := set(data).difference(self.vocab.get_stoi().keys())) != 0, \
            f"This dataset does not have the vocabulary required for training.\n The voc difference is : {d}"

        batch = wordsToOneHots(list(
            map(lambda w: '('*(self.n-1) + w + ')'*(self.n-1), data.split(" "))), self.vocab)

        # shape: ( T:=((L-self.n)/1)+1, B, self.n)
        batch_ngram = batch.unfold(0, self.n, 1)

        # shape: (T*B, self.n) ; shape: (*, self.n), (*)
        unique_ngrams, count_ngrams = torch.unique(
            batch_ngram.view(-1, self.n), sorted=False, return_counts=True, dim=0)

        non_zeros_ngram = torch.any(
            unique_ngrams[0] == 0, dim=1)     # shape: (*)

        return unique_ngrams.to(device), count_ngrams.to(device), non_zeros_ngram.to(device)

    def train(self, data: str) -> Tensor:
        """
        Train the n-gram language model on the `data` (str) 
        """
        training_data = self.prepare(data)

        for t, c, i in zip(*training_data):
            if i:
                self.nGramCount[tuple(t)] += c.item()

        # avoids all zeros cause it is the empty char so the prob in log is always null
        # for p in product(range(1, self.vocabSize), repeat=self.n):
        #     self.nGramCount[p] += torch.sum(
        #         torch.all(batch_ngram == torch.tensor(p), dim=-1)).item()

        countDivisor = torch.sum(self.nGramCount, dim=-1, keepdim=True)

        self.nGramLogProbs = torch.log(self.nGramCount/countDivisor)

        return self.nGramLogProbs

    def evaluation(self, data: str) -> float:
        """Perplexity evaluation"""
        # TODO: Perplexity
        return -1.0

    def padDataToNgram(self, reconstructions: Tensor) -> Tensor:
        """
        Pad reconstructions at the beginning and end of each sequence to conform to the calculation in ngram. 

        Args:
            reconstructions (Tensor, dim=(L,B)): input tensor to pad.

        Returns:
            (Tensor, dim=(L+2*(n-2), B)): padded reconstructions.
        """
        B = reconstructions.shape[1]

        first_padding = torch.full(
            (self.n-2, B), self.vocab['('], device=device)

        end_padding = torch.full(
            (self.n-2, B), self.vocab[PADDING_TOKEN], device=device)

        # shape: (L+2*(self.n-2), B)
        padded_tensor = torch.cat(
            (first_padding, reconstructions[0], end_padding), dim=0)

        # Indices of the first occurrence of 0 along the L dimension
        indices = torch.argmax(padded_tensor == self.vocab[PADDING_TOKEN],
                               dim=0)

        t = torch.arange(padded_tensor.shape[0], device=device).unsqueeze(1)
        mask = (t >= indices) & (t < indices +      # TODO: Check mask (dim)
                                 self.n-2)

        output = torch.where(mask, self.vocab[')'], padded_tensor)

        return output

    def inference(self, reconstructions: InferenceData) -> Tensor:
        data = self.padDataToNgram(reconstructions[0])
        maxSequenceLength, batch_size, V = data.shape
        # begin with the neutral prob 1
        probs: Tensor = torch.zeros(batch_size, dtype=torch.float64)

        # For each word by th in the batch, get the probability of the we sum (cause we
        # are in log) of all conditionnal probability of each character composed the word.
        for l in range(maxSequenceLength - self.n):
            nGram = data[l:l+self.n]
            nGramProb = self.nGramLogProbs.unsqueeze(
                0).expand(batch_size, *(-1,)*self.n)

            for k in range(self.n):
                selected_nGram = nGram[k]

                for _ in range(self.n-k-1):
                    selected_nGram = selected_nGram.unsqueeze(-1)

                selected_nGram = selected_nGram.expand(
                    -1, -1, *(V,)*(self.n-k-1))
                nGramProb = torch.logsumexp(selected_nGram + nGramProb, dim=1)

            # to neutralize the probability of ngrams with empty characters.
            probs += torch.min(nGramProb, torch.zeros(batch_size))
        return probs


class CharLM(nn.Module, PriorLM):
    """
    Character level language model.

    Params:
        embedding_size : Dimension of the character embedding vectors.
        hidden_size: Size of the LSTM hidden state.
        num_layers: Number of the layers of the LSTM.
        dropout_rate: Probability to drop out a neuron.    
        vocab (Vocab, optional) : The number of character in the vocabulary. Default value: `vocabulary`.
    """

    def __init__(self, embedding_size: int, hidden_size: int, num_layers: int, dropout_rate: float, vocab: Vocab = vocabulary):
        super(CharLM, self).__init__
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.embedding = PackingEmbedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.embedding_size, padding_idx=self.vocab[PADDING_TOKEN])

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=self.num_layers,
                            dropout=self.dropout_rate)

        self.fc = nn.Sequential(nn.Linear(self.hidden_size, self.vocab_size),
                                nn.LogSoftmax(dim=-1))

    def forward(self, x, h):
        embedded = self.embedding(x)
        output, h = self.lstm(embedded, h)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        output = self.fc(output)
        # Detach to avoid backpropagation through time
        return output, (h[0].detach(), h[1].detach())

    def init_hidden(self, batch_size: int, device=device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))

    def training(self, data: str, mini_batch_size: int = 32, epochs: int = 10, save_path: str = "./out/CharLM", learning_rate: float = 0.001):
        criter = nn.NLLLoss(ignore_index=self.vocab[PADDING_TOKEN])
        optim = Adam(self.parameters(), lr=learning_rate)

        indices_tensor = wordsToOneHots(data.split(' '), self.vocab)

        def adjust_seq_lengths(x: Tensor, l: Tensor, _): return (x, l+1)
        training_data = [adjust_seq_lengths(
            *computeInferenceData(tData)) for tData in indices_tensor.split(mini_batch_size, dim=1)]

        mini_batches_number = len(training_data)

        self.to(device)
        self.train()

        print('Training starts!')

        for epoch in tqdm(range(epochs)):
            total_loss = 0.0

            for (i, mini_training_data) in enumerate(training_data):
                optim.zero_grad()
                # MINI_BATCH_SIZE or batch_size % MINI_BATCH_SIZE
                # mini_batch_size = len(mini_training_data[1])
                hidden = self.init_hidden(mini_batch_size)

                scores, hidden = self(
                    mini_training_data[0], hidden)        # TODO: [:-1] ?
                trgt = mini_training_data[0][1:]

                loss = criter(scores.view(-1, self.vocab_size), trgt.view(-1))
                loss.backward()
                # nn.utils.clip_grad_norm_(self.parameters(), clip) ?
                optim.step()

                total_loss += loss.item()

                print(f'epoch: {epoch}/{epochs} ; \
                        step: {i + 1}/{mini_batches_number} ; \
                        loss: {loss.item()}')

            average_loss = total_loss / mini_batches_number
            print(f'epoch: {epoch}/{epochs} ; average loss: {average_loss}')
            torch.save(self.state_dict(), f'{save_path}_{epoch}.pt')

        print('Training ends!')

    def evaluation(self) -> float:
        # TODO: Perplexity
        return -1.0

    def inference(self, reconstructions: InferenceData) -> Tensor:
        self.eval()

        with torch.no_grad():
            eval_data = reconstructions[0].to(device)
            seq_length, batch_size = eval_data.shape

            hidden = self.init_hidden(batch_size)

            output, _ = self(eval_data, hidden)  # shape (L, B, output_dim)

            # TODO: fix the index error for padding tokens (for which the one hot index will be out of range for the distribution)
            output = output[torch.arange(seq_length-1).unsqueeze(1), torch.arange(
                batch_size).unsqueeze(0), eval_data[1:]]  # shape (L-1, B)
            output = torch.sum(output, dim=0)  # shape (B)

        return output
