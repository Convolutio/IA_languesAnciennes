import torch
from torch import Tensor
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torch import cuda
import re

from models.types import ModernLanguages, InferenceData, InferenceData_Cognates, InferenceData_Samples, SOS_TOKEN, EOS_TOKEN, PADDING_TOKEN

device = "cuda" if cuda.is_available() else "cpu"

__specialTokensPattern = re.compile('|'.join(
    ['(' + specialToken + ')' for specialToken in (PADDING_TOKEN, SOS_TOKEN, EOS_TOKEN)]))

vocabulary: Vocab  # |Σ| = len(vocab)-3
with open('./data/IPA_vocabulary.txt', 'r', encoding='utf-8') as vocFile:
    vocabulary = build_vocab_from_iterator(vocFile.read().split(
        ", "), specials=[EOS_TOKEN, SOS_TOKEN, PADDING_TOKEN], special_first=False)
IPA_charsNumber = len(vocabulary)-3


def wordsToOneHots(words: list[str], inventory: Vocab = vocabulary) -> Tensor:
    """
    Args:
        words (list[str]): list of words to convert.
        inventory (Vocab, optional): vocabulary (object) for mapping characters into integers. Default value: `vocabulary`.

    Returns:
        IntTensor, dim=(L, B): a padded tensor containing the B words of the list as sequences of L tokens in one-hot indices format.

    Examples:
    >>> wordsToOneHots(['notifikar', 'dɔrikʊ']).T
    tensor([[11, 12, 16,  6,  4,  6,  8,  0, 14],
        [ 2, 30, 14,  6,  8, 43, 59, 59, 59]], dtype=torch.int32)
    """
    return pad_sequence([torch.IntTensor(inventory(list(word))) for word in words], batch_first=False, padding_value=inventory[PADDING_TOKEN]).to(device=device)


def paddedOneHotsToRawSequencesList(batch: InferenceData) -> list[Tensor]:
    """
    Args:
        batch (InferenceData): the first element must be a padded IntTensor of shape (L, B).

    Returns:
        list[Tensor]: a list of unidimensional IntTensors of variable length.
    """
    return unpad_sequence(batch[0][1:], batch[1], batch_first=False)


def oneHotsToWords(batch: Tensor, removeSpecialTokens: bool = False, inventory: Vocab = vocabulary) -> list[str]:
    """
    Args:
        batch (Tensor, dim=(L, B)): IntTensor/LongTensor representing padded words in sequences of one hot indices.
        removeSpecialTokens (bool, optional): argument to remove or not the special tokens. Default value: `False`.
        inventory (Vocab, optional): vocabulary (object) for mapping indices into characters. Default value: `vocabulary`.

    Returns:
        list[str]: a list of B words in string format.

    Examples:
    >>> t = tensor([[58, 11, 12, 16,  6,  4,  6,  8,  0, 14, 57],
        [58,  2, 30, 14,  6,  8, 43, 57, 59, 59, 59]], dtype=torch.int32).T
    >>> t.size()
    torch.Size([11, 2])
    >>> oneHotsToWords(t, False, vocabulary)
    ['(notifikar)', '(dɔrikʊ)---']
    >>> oneHotsToWords(t, True, vocabulary)
    ['notifikar', 'dɔrikʊ']
    """
    wordsLst = ["".join(inventory.lookup_tokens(wordInLst))
                for wordInLst in batch.T.tolist()]
    if not removeSpecialTokens:
        return wordsLst
    else:
        return __specialTokensPattern.sub("", " ".join(wordsLst)).split(" ")


def __computeInferenceData(words_intTensor: Tensor, vocab: Vocab = vocabulary) -> InferenceData:
    """
    Computes data for the inference from an IntTensor containing words in one-hot indices format.
    The byteTensor is reduced, then the lengths of the sequences are computed and finally
    the one-hot vector encoding is performed.

    Args:
        words_intTensor (ByteTensor, dim=(ArbitrarySequenceLength, *)) : tensor with the encoded words.
        vocab (Vocab, optional): vocabulary containing the mapping between characters and indices. Default value: `vocabulary`.

    Returns:
        InferenceData: compute data for the inference. 
    """
    left_boundary_index = vocab[SOS_TOKEN]
    right_boundary_index = vocab[EOS_TOKEN]
    padding_index = vocab[PADDING_TOKEN]

    rawShape = words_intTensor.size()

    withBoundariesTensor = torch.cat((
        torch.full((1, *rawShape[1:]), left_boundary_index,
                   device=device, dtype=torch.int32),
        words_intTensor,
        torch.full((1, *rawShape[1:]), padding_index,
                   device=device, dtype=torch.int32)
    ))

    t = torch.logical_xor(
        withBoundariesTensor[:-1] != padding_index, withBoundariesTensor[1:] != padding_index)

    withBoundariesTensor[1:] = torch.where(
        t, right_boundary_index, withBoundariesTensor[1:])

    lengths = torch.argmax(t.to(torch.uint8), dim=0) + 2
    maxLength = int(torch.max(lengths).item())

    return (withBoundariesTensor[:maxLength], lengths.cpu(), maxLength)

def computeInferenceData_Samples(words_intTensor: Tensor, vocab: Vocab = vocabulary) -> InferenceData_Samples:
    """
    Arguments:
        - words_intTensor: an IntTensor of shape (|x|, c) or (|x|, c, b)
    Computes samples' input data for the ReconstructionModel from an IntTensor containing words in one-hot indices format.
    The IntTensor is reduced, then the lengths of the sequences are computed and finally the encoding is performed.
    The one-hot indices IntTensor is of shape (|x|+2, c, b) and the lengths cpu Tensor is of shape (c, b)
    """
    dimensions_number = len(words_intTensor.size())
    if dimensions_number == 2:
        words_intTensor = words_intTensor.unsqueeze(-1)
    elif dimensions_number != 3:
        raise Exception("Your tensor with samples' tokens must be of shape (|x|, c) or (|x|, c, b)")
    return __computeInferenceData(words_intTensor, vocab)

def computeInferenceData_Cognates(words_intTensors: dict[ModernLanguages, Tensor], vocab: Vocab = vocabulary) -> dict[ModernLanguages, InferenceData_Cognates]:
    """
    Arguments:
        - words_intTensor: a dictionnary of IntTensor of shape (|y_l|, c)
    Computes cognates' input data for the ReconstructionModel from IntTensors containing words in one-hot indices format.
    The IntTensors are reduced, then the lengths of the sequences are computed and finally the encoding is performed.
    The EOS token is removed and so the lengths Tensor equals (|y|+1). See the `InferenceData_Cognates` type's documentation for more details.
    """
    d:dict[ModernLanguages, InferenceData_Cognates] = {}
    for lang in words_intTensors:
        targets, rawLengths, maxLength = __computeInferenceData(words_intTensors[lang], vocab)
        d[lang] = (targets.where(targets != vocab[EOS_TOKEN], vocab[PADDING_TOKEN])[:-1],
            rawLengths-1, maxLength-1)
    return d