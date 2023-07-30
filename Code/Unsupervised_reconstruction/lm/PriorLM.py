from itertools import permutations

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm.auto import tqdm

from Types.models import InferenceData
from data.vocab import wordToOneHots, reduceOneHotTensor, make_oneHotTensor, SIGMA

INFTY_NEG = -1e9
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PriorLM:
    def __init__(self):
        pass

    def train(self, data: str) -> None: ...

    def evaluation(self) -> float: ...

    def inference(self, reconstructions: InferenceData) -> Tensor: ...


class NGramLM(PriorLM):
    def __init__(self, n: int, vocab: str, boundaries: str = "()", device=device):
        self.device = device
        self.n: int = n
        self.vocab: dict[str, int] = {
            c: i+1 for i, c in enumerate("" + vocab + boundaries)}  # ...+1 cause empty char is 0
        self.vocabSize: int = len(vocab) + len(boundaries) + 1

        self.nGramCount: Tensor = torch.zeros(
            (self.vocabSize,) * self.n, device=device)
        self.nGramLogProbs: Tensor = torch.log(
            torch.zeros((self.vocabSize,) * self.n, device=device))

    @staticmethod
    def countSubtensorOccurrences(larger_tensor: Tensor, sub_tensor: Tensor) -> int:
        # TODO: Removes (deprecated)
        """
        Count the number of sub-tensors that can exist in a large tensor (n-gram batch).

        *This function was designed to count n-grams in a large tensor of size (L x B x V),
        each row (L) of which corresponds to a character in a word in a column (B),
        and the depth (V) of which corresponds to the one-hot tensor associated with that character.*

        >>> bigTensor = torch.tensor([[[0,0,1], [0,1,0], [0,0,1]],
                                        [[0,0,1], [0,1,0], [0,0,0]]])
        >>> littleTensor = torch.tensor([[0,0,1], [0,1,0]])
        >>> countSubtensorOccurrences(bigTensor, littleTensor)
        2
        """

        # Forms the n-grams.
        larger_reshaped = larger_tensor.unfold(
            1, sub_tensor.size()[0], 1).unfold(2, sub_tensor.size()[1], 1)

        t = torch.all(larger_reshaped == sub_tensor, dim=2)
        mask = torch.all(t, dim=3)
        t_masked = t[mask]  # To keep only full True one hot character tensor.

        occurrences = torch.sum(t_masked).item(
        )//(sub_tensor.size()[0]*sub_tensor.size()[1])

        return int(occurrences)

    def prepare(self, data: str) -> Tensor:
        """
        Prepare the data in a batch by checking the validity \
        of the vocabulary and setting the tensor to the shape (L x B).

        Args:
            data (str): the training data 

        Returns:
            Tensor (shape: L x B)
        """
        assert len(d := set(data).difference(self.vocab.keys())) != 0, \
            f"This dataset does not have the vocabulary required for training.\n The voc difference is : {d}"

        wordsList = data.split(" ")

        batch = reduceOneHotTensor(pad_sequence(
            [wordToOneHots('('*(self.n-1) + w + ')'*(self.n-1), self.vocab) for w in wordsList], batch_first=False))  # shape: (L x B)

        return batch.to(device)

    def train(self, data: str) -> Tensor:
        """
        Train the n-gram language model on the `data` (str) 
        """
        batch = self.prepare(data)
        batch_ngram = batch.unfold(
            1, self.vocabSize, 1)  # TODO: Check dim
        batch_ngram_id = torch.sum(batch_ngram, dim=-1)  # TODO: Check dim

        # avoids all zeros cause it is the empty char so the prob in log is always null
        for i, idx in enumerate(permutations(range(1, self.vocabSize), self.n)):
            self.nGramCount[idx] = torch.sum(
                batch_ngram_id == i).item()  # += or = ?

        countDivisor = torch.sum(
            self.nGramCount, dim=-1).expand(self.nGramCount.shape)

        # TODO: Check division (especially -inf values)
        self.nGramLogProbs = torch.log(self.nGramCount/countDivisor)

        return self.nGramLogProbs

    def evaluation(self, data: str) -> float:
        """Perplexity evaluation"""
        # TODO: Perplexity
        return -1.0

    def padDataWithN(self, reconstructions: InferenceData) -> Tensor:
        # TODO: Remove (Deprecated)! 
        """
        Returns the reconstructions in one-hot vectors tensor format, with ensuring that sequences with a length smaller than n are padded with the ')' token.

        Args:
            reconstructions (InferenceData) : a tuple whose the first term is a tensor of (L, B, V) shape.
        """
        # ngrams with empty characters have a neutral probability of 1.
        data, lengths = nn.utils.rnn.pad_packed_sequence(
            reconstructions[0], padding_value=1.0)  # type:ignore (Tensor!=PackedSequence)
        data, lengths = data.to(device), lengths.to(device)
        maxSequenceLength, batch_size, V = data.shape

        if maxSequenceLength < self.n:
            data = torch.cat((data, torch.zeros(
                (self.n - maxSequenceLength, batch_size, V), device=device)), dim=0)
            maxSequenceLength = self.n

        a = torch.arange(maxSequenceLength).to(
            device).unsqueeze(0).expand(batch_size, -1)

        condition = torch.logical_and(lengths.unsqueeze(
            1).expand(-1, maxSequenceLength) <= a, a < self.n).T.unsqueeze(-1).expand(-1, -1, V)  # size = (L, B, V)

        pad = torch.zeros((maxSequenceLength, batch_size, V),
                          dtype=torch.float32, device=device)
        pad[:, :, V-1] = 1

        data = torch.where(condition, pad, data)

        return torch.log(data)  # Why log ? Data aren't already in log ?

    def padDataToNgram(self, reconstructions: Tensor) -> Tensor:
        """
        Returns the padded reconstructions at the beginning and end of each sequence to conform to the calculation in ngram. 
        The final shape of the tensor is (L+2*(n-2), B, V)

        Args:
            reconstructions (Tensor) : tensor of (L, B, V) shape.
        """

        # TODO: Remove the V dimension

        L, B, V = reconstructions.shape

        first_padding = torch.full(
            (self.n-2, B, V), self.vocab['('], device=device)
        # extra 0 padding for torch.where
        end_padding = torch.full((self.n-2, B, V), 0, device=device)

        # shape: (L+2*(self.n-2), B, V)
        padded_tensor = torch.cat(
            (first_padding, reconstructions[0], end_padding), dim=0)

        # Indices of the first occurrence of 0 along the L dimension
        indices = torch.argmax(padded_tensor == 0,
                               dim=0)  # TODO: Check validity

        t = torch.arange(padded_tensor.shape[0], device=device).unsqueeze(1)
        mask = (t >= indices) & (t < indices +
                                 self.n-2)  # TODO: Check validity

        output = torch.where(mask, self.vocab[')'], padded_tensor)

        return output.to(device)

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
        return probs.to(device)


class CharLMDataset(Dataset):
    def __init__(self, data: list[str], vocab: dict):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.word_to_one_hot(self.data[idx], self.vocab)

    @staticmethod
    def word_to_one_hot(word: str, vocab: dict[str, int] = SIGMA) -> Tensor:
        """
        Convert a word into a one-hot tensor using a predefined vocabulary.

        Args:
            word (str): The input word to convert.
            vocab (dict): A dictionary that maps characters to their corresponding indices in the one-hot vector.

        Returns:
            Tensor: A one-hot tensor representation of the input word.
        """
        one_hot = torch.zeros(len(word), len(vocab))
        for i, char in enumerate(word):
            char_index = vocab[char]
            one_hot[i][char_index] = 1
        return one_hot.to(device)


class CharLM(nn.Module, PriorLM):
    """
    Character level language model.

    Params:
        vocab_size (int) : The number of character in the vocabulary.
        embedding_size : Dimension of the character embedding vectors.
        hidden_size: Size of the LSTM hidden state.
        num_layers: Number of the layers of the LSTM.
        dropout_rate: Probability to drop out a neuron.    
    """

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, num_layers: int, dropout_rate: float):
        super(CharLM, self).__init__
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embedding_size)

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=self.num_layers,
                            dropout=self.dropout_rate)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, h):
        embedded = self.embedding(x)
        output, h = self.lstm(embedded, h)
        output = self.fc(output)
        # Detach to avoid backpropagation through time
        return output, (h[0].detach(), h[1].detach())

    def init_hidden(self, batch_size: int, device=device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))

    def prepare(self, data: str) -> tuple[int, tuple[Tensor, Tensor]]:
        """
        Prepare the data for the character level language model training.

        Args:
            data (str): The data to prepare.

        Returns:
            tuple[int, tuple[Tensor, Tensor]]: The batch size with the input and target tensors. 
        """
        # Split by space
        # TODO: Add boundaries for each word ?
        words = data.split(' ')
        batch_size = len(words)

        # Convert each word into a one-hot (binary) tensor
        one_hot_tensors = [CharLMDataset.word_to_one_hot(word) for word in words]     # TODO: Vocab?

        # Pad the tensor (it will return a tensor of LxBxV shape)
        padded_tensor = pad_sequence(one_hot_tensors, batch_first=False)

        # Create the training dataset (not memory healthy, but ok cause the dataset is pretty small)
        # (input_tensor, target_tensor) with a shape of (L-1)xBxV
        training_data = (padded_tensor[1:, :, :].to(device),
                         padded_tensor[:-1, :, :].to(device))

        return batch_size, training_data

    def training(self, data: str, epochs: int, save_path: str, learning_rate: float = 0.001):
        criter = nn.CrossEntropyLoss()
        optim = Adam(self.parameters(), lr=learning_rate)

        batch_size, training_data = self.prepare(data)

        self.to(device)
        self.train()  # TODO: Correct this line / Train mode

        print('Training starts!')
        for epoch in tqdm(range(epochs)):
            total_loss = 0.0
            hidden = self.init_hidden(batch_size)

            for i in range(batch_size):
                # TODO: Can optimise this!    Loop over string ; transform at the moment in one hot
                inpt, trgt = training_data[0][:, i, :], training_data[1][:, i, :]
                optim.zero_grad()

                scores, hidden = self(inpt, hidden)

                loss = criter(scores.view(-1, self.vocab_size), trgt.view(-1))      # TODO: Check if it is the good dim
                loss.backward()
                # nn.utils.clip_grad_norm_(self.parameters(), clip)
                optim.step()

                total_loss += loss.item()

                print(f'epoch: {epoch}/{epochs} ; \
                        step: {i + 1}/{batch_size} ; \
                        loss: {loss.item()}')

            average_loss = total_loss / batch_size
            print(f'epoch: {epoch}/{epochs} ; average loss: {average_loss}')
            torch.save(self.state_dict(), save_path)        # TODO: Correct save path

        print('Training ends!')

    def evaluation(self) -> float:
        # TODO: Perplexity
        return -1.0

    # TODO
    def indicesToOneHot(self, t: Tensor) -> Tensor: ...

    def inference(self, reconstructions: InferenceData) -> Tensor:
        self.eval()

        with torch.no_grad():
            #TODO: Convert indices reconstructions into one hot reconstructions
            eval_data: Tensor = self.indicesToOneHot(reconstructions[0]).to(device)
            seq_length, batch_size, vocab_size = eval_data.shape

            hidden = self.init_hidden(batch_size)

            log_probs = torch.zeros(seq_length, batch_size, vocab_size, device=device)

            for i in range(batch_size):
                inpt = eval_data[:, i, :]
                output, hidden = self(inpt, hidden)
                log_probs_t = nn.functional.log_softmax(output, dim=1)
                log_probs[:, i, :] = log_probs_t

        return log_probs
