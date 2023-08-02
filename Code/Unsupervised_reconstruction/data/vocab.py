import torch
from torch import Tensor
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torch import cuda
import re

from Types.models import InferenceData, SOS_TOKEN, EOS_TOKEN, PADDING_TOKEN

device = "cuda" if cuda.is_available() else "cpu"

specialTokensPattern = re.compile('|'.join(['(' + specialToken + ')' for specialToken in (PADDING_TOKEN, SOS_TOKEN, EOS_TOKEN)]))

vocabulary:Vocab # |Σ| = len(vocab)-3
with open('./data/IPA_vocabulary.txt', 'r', encoding='utf-8') as vocFile:
    vocabulary = build_vocab_from_iterator(vocFile.read().split(", "), specials=[EOS_TOKEN, SOS_TOKEN, PADDING_TOKEN], special_first=False)
IPA_charsNumber = len(vocabulary)-3

def wordsToOneHots(words: list[str], inventory: Vocab = vocabulary) -> torch.Tensor:
    """
    Arguments:
        - words (list[str]): a list of words
        - inventory (Vocab): the vocabulary object to do the mapping
    Returns a padded IntTensor of shape (L, B) containing the B words in the list as sequences of L tokens in one-hot indices format.

    >>> wordsToOneHots(['notifikar', 'dɔrikʊ']).T
    tensor([[11, 12, 16,  6,  4,  6,  8,  0, 14],
        [ 2, 30, 14,  6,  8, 43, 59, 59, 59]], dtype=torch.int32)
    """
    return pad_sequence([torch.IntTensor(inventory(list(word))) for word in words], batch_first=False, padding_value=inventory[PADDING_TOKEN]).to(device=device)

def paddedOneHotsToRawSequencesList(batch:InferenceData) -> list[Tensor]:
    """
    Arguments:
        - batch (InferenceData): the first element must be a padded IntTensor of shape (L, B)
    Returns a list of unidimensional IntTensors of variable length.
    """
    return unpad_sequence(batch[0][1:], batch[1], batch_first=False)

def oneHotsToWords(batch: Tensor, removeSpecialTokens:bool = False, inventory: Vocab = vocabulary, ):
    """
    Arguments:
        - batch (Tensor): an IntTensor/LongTensor representing padded words in sequences of one hot indexes.
        dim = (L, B)
    Returns a list of B words in string format.

    >>> t = tensor([[58, 11, 12, 16,  6,  4,  6,  8,  0, 14, 57],
        [58,  2, 30, 14,  6,  8, 43, 57, 59, 59, 59]], dtype=torch.int32).T
    >>> t.size()
    torch.Size([11, 2])
    >>> oneHotsToWords(t, vocabulary)
    ['(notifikar)', '(dɔrikʊ)---']
    >>> oneHotsToWords(t, vocabulary, True)
    ['notifikar', 'dɔrikʊ']
    """
    wordsLst = ["".join(inventory.lookup_tokens(wordInLst)) for wordInLst in batch.T.tolist()]
    if not removeSpecialTokens:
        return wordsLst
    else:
        return specialTokensPattern.sub("", " ".join(wordsLst)).split(" ")

def computeInferenceData(words_intTensor: Tensor, vocab:Vocab = vocabulary) -> InferenceData:
    """
    Computes data for the inference from an IntTensor containing words in one-hot indexes format.
    To do that, the byteTensor is reduced, then the lengths of the sequences are computed and finally
    the one-hot vector encoding is carried out.

    Arguments:
        byteTensor (ByteTensor, dim=(ArbitrarySequenceLength, *)) : the tensor with the encoded words.  
    """
    left_boundary_index = vocab['(']
    right_boundary_index = vocab[')']
    padding_index = vocab[PADDING_TOKEN]

    rawShape = words_intTensor.size()
    withBoundariesTensor = torch.cat((
            torch.full((1, *rawShape[1:]), left_boundary_index, device=device, dtype=torch.int32),
            words_intTensor,
            torch.full((1, *rawShape[1:]), padding_index, device=device, dtype=torch.int32)
        ))
    t = torch.logical_xor(withBoundariesTensor[:-1] != padding_index, withBoundariesTensor[1:] != padding_index)
    withBoundariesTensor[1:] = torch.where(t, right_boundary_index, withBoundariesTensor[1:])
    lengths = torch.argmax(t.to(torch.uint8), dim=0)
    maxLength = int(torch.max(lengths).item())

    return (withBoundariesTensor[:maxLength+2], lengths.cpu(), maxLength)

