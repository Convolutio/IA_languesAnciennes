# Sources :
# https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
# https://towardsdatascience.com/language-modeling-with-lstms-in-pytorch-381a26badcbf
# https://www.youtube.com/watch?v=euwN5DHfLEo

import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torch.optim import AdamW, optim, sgd

from ipa_tokenizer import tokenize_ipa


"""
PARAMETERS
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# Define the hyperparameters
# vocab_size = 27
# embedding_size = 128
# hidden_size = 512
# num_layers = 2
# dropout = 0.2
# learning_rate = 0.001
# batch_size = 64
# num_epochs = 20


"""
DATASET CLASS
"""

class CharacterDataset(Dataset):
    """
        Custom Dataset at character-level (IPA).

        Params:
            - `raw_data` (str) :
                Input data will be used to create the dataset.

            - `vocab` (list[str]) :
                Vocabulary. Set of characters allowing the tokenization of input data and defines the characters taken into account or not.

            - `oov` (str) :
                Out of vocabulary. All characters outside the vocabulary will be assigned to this string at tokenisation.

            - `batch_size` (int) :
                Number of characters in a sequence.

        Attrs:
            - `char2idx` (defaultdict) :
                Character to index.
                Mapping from the character (char ; c) to its index (idx) in the vocabulary.
                Note that a character that is not in the vocabulary will get mapped into the index `vocab_size`.

            - `idx2char` (dict) :
                Index to character.
                Reverse of char2idx.

            - `tokens` (list[str]) :
                Tokenized input data.
                Each element in the list correspond to an IPA character.
    """

    def __init__(self, raw_data: str, vocab: list[str], oov: str, batch_size: int):
        self.batch_size = batch_size    # or sequence length

        # Create the mapping dictonary
        # 0 -> len(vocab)-1 are dedicated for el in `vocab`
        # len(vocab) or last el is dedicated for the `oov`
        self.char2idx = defaultdict(lambda: len(vocab))
        self.char2idx.update({el: i for i, el in enumerate(vocab)})
        self.char2idx[oov] = len(vocab)

        # Create the reverse dictionary
        self.idx2char = {idx: char for idx,
                               char in self.char2idx.items()}

        # Tokenize the raw_data by IPA characters.
        self.tokens: list[str] = self.tokenize_ipa(raw_data)    # TODO: Import the lib

        # Removed for memory optimisation !
        # self.tokens_idx: list[str] = [self.char2idx[char]
        #                              for char in self.tokens]

    def __len__(self):
        return len(self.tokens) - self.batch_size

    def __getitem__(self, index: int):
        return (torch.tensor([self.char2idx[c] for c in self.tokens[index:index+self.batch_size]]),
                torch.tensor([self.char2idx[c] for c in self.tokens_idx[index+1:index+self.batch_size+1]]))     # TODO: Check if it is not out of index


"""
LM CLASS
"""

class RNNLM(nn.Module):
    """
    Classic network predicting the next character of a string.

    Params:
        - `vocab_size` (int) :
            The number of character in the vocabulary.

        - `embedding_size` (int) :
            Dimension of the character embedding vectors.

        - `hidden_size`:
            Size of the LSTM hidden state.

        - `num_layers`:
            Number of the layers of the LSTM.

        - `dropout_rate`:
            Probability to drop out a neuron.    
    """

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, num_layers: int, dropout_rate: float):
        super(RNNLM, self).__init__()
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


"""
TRAINING
"""
# TODO: Training ?


"""
EVALUATION
"""
# TODO: Perplexity ?


"""
PREDICT
"""
# TODO: ?