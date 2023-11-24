import torch
import torch.nn as nn

from torch import Tensor
from torch.types import Device
from torchtext.vocab import Vocab
from torch.optim import Adam
from tqdm.auto import tqdm

from models.types import InferenceData_Samples, PADDING_TOKEN, SOS_TOKEN, EOS_TOKEN
from data.vocab import wordsToOneHots, computeInferenceData_Samples, vocabulary
from Source.packingEmbedding import PackingEmbedding

from typing import Callable


class PriorLM:
    def __init__(self): ...
    def train(self, data: str) -> None: ...
    def evaluation(self) -> float: ...
    def inference(self, reconstructions: InferenceData_Samples) -> Tensor: ...


class NGramLM(PriorLM):
    """
    ## N-gram Language Model.

    Params:
        n (int): size of the N-gram to be used in the model.
        vocab (Vocab, optional): vocabulary of IPA characters in the language model.
        smoothingValue (float, optional): value used for smoothing in probability computations for n-grams and (n-1)-grams not encountered.
                                Defaults value: -1e5.
    """

    def __init__(self, n: int, device: Device, vocab: Vocab = vocabulary, smoothingValue: float = -1e5):
        self.device = torch.device(f"{device}")
        
        self.n = n
        self.vocab = vocab
        self.vocabSize = len(vocab)
        self.smoothingValue = smoothingValue

        self.nGramCount = torch.zeros(
            (self.vocabSize,) * self.n, dtype=torch.float64, device=self.device)
        self.nGramLogProbs = torch.log(
            torch.zeros((self.vocabSize,) * self.n, dtype=torch.float64, device=self.device))

    def batch_ngram(self, data: str) -> Tensor:
        """
        Prepare string data into an ngram batch tensor while checking the validity of the vocabulary.

        Args:
            data (str): input training data.

        Returns:
            Tensor, dim=(*, B, self.n): ngram batch tensor.
        """

        d = set(data)
        d.discard(' ')
        if len(d.difference(self.vocab.get_stoi().keys())) != 0:
            raise Exception(
                f"This dataset does not have the vocabulary required for training.\n The voc difference is : {d}")
        del d

        # Before converting the string of words into a padded tensor,
        # it is necessary to convert it into a list and add the SOS token and the EOS token to each word.
        addBoundaries = lambda w: SOS_TOKEN*(self.n-1) + w + EOS_TOKEN*(self.n-1)
        batch = wordsToOneHots(list(map(addBoundaries, data.split(' '))), 
                               self.device, self.vocab)

        # shape: ( T:=((L-self.n)/1)+1, B, self.n)
        batch_ngram = batch.unfold(0, self.n, 1)

        return batch_ngram

    def padDataToNgram(self, reconstructions: Tensor) -> Tensor:
        """
        Pad reconstructions (with boundaries) at the beginning and end of each sequence to conform to the computation in ngram. 

        Args:
            reconstructions (dim=(L,*)): input tensor to pad.

        Returns:
            Tensor, dim=(L+2*(n-1), *): padded reconstructions.
        """
        # This method allows the L dimension to be reworked,
        # so that the other dimensions are kept in their original state.
        batchShape = reconstructions.shape[1:]

        #TODO: Boundaries or not?
        # if torch.any(reconstructions == self.vocab[SOS_TOKEN]) or torch.any(reconstructions == self.vocab[EOS_TOKEN]):
        #     raise Exception("There are SOS and/or EOS tokens in the reconstruction tensor.")

        first_padding = torch.full(
            (self.n-2, *batchShape), self.vocab[SOS_TOKEN], device=self.device)

        # Dedicate the space for the (n-2) EOS to be placed at the end of the words.
        end_padding = torch.full(
            (self.n-2, *batchShape), self.vocab[PADDING_TOKEN], device=self.device)

        # shape: (L+2*(self.n-2), *)
        padded_tensor = torch.cat(
            (first_padding, reconstructions, end_padding), dim=0)

        # Indices of the first occurrence of PADDING_TOKEN along the L dimension
        indices = torch.argmax((padded_tensor == self.vocab[PADDING_TOKEN])*1,
                               dim=0)

        t = torch.arange(padded_tensor.shape[0], device=self.device)
        t = t.reshape(((t.shape[0],)+len(batchShape)*(1,)))
        mask = (t >= indices) & (t < indices + self.n-2)

        output = torch.where(mask, self.vocab[EOS_TOKEN], padded_tensor)

        return output

    def train(self, data: str) -> Tensor:
        """
        Train the n-gram language model on the `data` (str).
        """
        batch_ngram = self.batch_ngram(data)

        # batch_ngram.view shape: (T*B, self.n) ; unique_ngrams shape: (*, self.n); count_ngrams shape (*)
        unique_ngrams, count_ngrams = torch.unique(
            batch_ngram.view(-1, self.n), sorted=False, return_counts=True, dim=0)

        non_padded_ngram = torch.all(
            unique_ngrams != self.vocab[PADDING_TOKEN], dim=1)     # shape: (*)

        count_ngrams = torch.where(non_padded_ngram, count_ngrams, 0).to(
            dtype=self.nGramCount.dtype)
        coords: tuple[Tensor, ...] = tuple(
            unique_ngrams.T)  # coord tensor shape = (*)
        self.nGramCount[coords] = count_ngrams

        countDivisor = torch.sum(self.nGramCount, dim=-1, keepdim=True)

        # smooth the probs for non met ngrams
        # TODO: smoothing with better algorithm
        positionToSmooth = torch.logical_or(
            countDivisor == 0, self.nGramCount == 0)

        self.nGramLogProbs = torch.where(positionToSmooth,
                                         self.smoothingValue,
                                         torch.log(self.nGramCount) -
                                         torch.log(countDivisor))

        # neutralize undefined ngrams
        for k in range(self.n):
            coord = (None,)*k + \
                (self.vocab[PADDING_TOKEN],) + (None,)*(self.n-k-1)
            self.nGramLogProbs[coord] = -1e9

        return self.nGramLogProbs

    def evaluation(self, data: str) -> float:
        """Perplexity evaluation"""
        batch_ngram = self.batch_ngram(data)
        #TODO
        return -1.0

    def inference(self, reconstructions: InferenceData_Samples) -> Tensor:
        data = self.padDataToNgram(reconstructions[0])

        """
        Let (L, *) be the shape of data
        Coords is a tuple of n tensors of shape (*..., (L-n)/1 + 1)
        """
        coords = tuple(data.unfold(0, self.n, 1).transpose(0, -1))
        probs = self.nGramLogProbs[coords].sum(dim=-1)  # shape = (*)
        return probs


class CharLM(nn.Module, PriorLM):
    """
    ## RNN sound-level language model.

    Params:
        embedding_size : dimension of the character embedding vectors.
        hidden_size: size of the LSTM hidden state.
        num_layers: number of the layers of the LSTM.
        dropout_rate: probability to drop out a neuron. 
        device: device where the model be put on.  
        vocab (Vocab, optional) : number of character in the vocabulary. Default value: `vocabulary`.
    """

    def __init__(self, embedding_size: int, hidden_size: int, num_layers: int, dropout_rate: float, device, vocab: Vocab = vocabulary):
        super(CharLM, self).__init__()
        self.device = device
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.output_dim = self.vocab_size - 2
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.embedding = PackingEmbedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.embedding_size, padding_idx=self.vocab[PADDING_TOKEN])

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=self.num_layers,
                            dropout=self.dropout_rate)

        self.fc = nn.Sequential(nn.Linear(self.hidden_size, self.output_dim),
                                nn.LogSoftmax(dim=-1))

    def __call__(self, x: tuple[Tensor, Tensor], h: tuple[Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        return super().__call__(x, h)

    def forward(self, x: tuple[Tensor, Tensor], h: tuple[Tensor, Tensor]):
        embedded = self.embedding((x[0], x[1], False))
        output, h = self.lstm(embedded, h)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        output = self.fc(output)
        # Detach to avoid backpropagation through time
        return output, (h[0].detach(), h[1].detach())

    def init_hidden(self, batch_size: int):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))

    def training(self, data: str, mini_batch_size: int = 32, epochs: int = 10, save_path: str = "./out/CharLM", learning_rate: float = 1e-3):
        self.to(self.device) #TODO: Check if necessary
        self.train()

        criter = nn.NLLLoss(ignore_index=self.vocab[PADDING_TOKEN])
        optim = Adam(self.parameters(), lr=learning_rate)

        indices_tensor = wordsToOneHots(data.split(' '), self.device, self.vocab)

        adjust_seq_lengths: Callable[[Tensor, Tensor, int],
                                     tuple[Tensor, Tensor]] = lambda x, l, _: (x, l+1)
        training_data = [adjust_seq_lengths(
            *computeInferenceData_Samples(tData)) for tData in indices_tensor.split(mini_batch_size, dim=1)]

        mini_batches_number = len(training_data)

        print('Training starts!')

        for epoch in tqdm(range(epochs)):
            total_loss = 0.0

            for (i, mini_training_data) in enumerate(training_data):
                optim.zero_grad()
                # MINI_BATCH_SIZE or batch_size % MINI_BATCH_SIZE
                # mini_batch_size = len(mini_training_data[1])
                hidden = self.init_hidden(mini_batch_size)

                # scores shape = (|x|+1, b, output_dim)
                scores, hidden = self(
                    mini_training_data, hidden)
                trgt = mini_training_data[0][1:]  # shape = (|x|+1, b)

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
        # https://stackoverflow.com/questions/59209086/calculate-perplexity-in-pytorch
        return -1.0

    def inference(self, reconstructions: InferenceData_Samples) -> Tensor:
        self.eval()

        with torch.no_grad():
            neutralizePaddingIndices: Callable[[Tensor], Tensor] = lambda t: t.where(
                t != self.vocab[PADDING_TOKEN], self.output_dim)
            addNeutralProbsForPaddingIndices: Callable[[Tensor], Tensor] = lambda t: torch.cat(
                (t, torch.zeros(t.size()[:-1]+(1,),
                 dtype=t.dtype, device=self.device)),
                dim=-1
            )
            eval_data = neutralizePaddingIndices(reconstructions[0])
            seq_length, batch_size = eval_data.shape

            hidden = self.init_hidden(batch_size)

            output = addNeutralProbsForPaddingIndices(self(
                (reconstructions[0], reconstructions[1]+2), hidden)[0])  # shape (L, B, output_dim + 1)

            output = output[torch.arange(seq_length-1).unsqueeze(1), torch.arange(
                batch_size).unsqueeze(0), eval_data[1:]]  # shape (L-1, B)
            output = torch.sum(output, dim=0)  # shape (B)

        return output