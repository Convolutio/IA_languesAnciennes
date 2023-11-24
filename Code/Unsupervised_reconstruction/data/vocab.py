import re
import torch
import random

from typing import Optional
from torch.types import Device
from torch import Tensor

from torchtext.vocab import Vocab, build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from models.types import ModernLanguages, InferenceData, InferenceData_Cognates, InferenceData_Samples, SOS_TOKEN, EOS_TOKEN, PADDING_TOKEN


vocabulary: Vocab
with open('./data/IPA_vocabulary.txt', 'r', encoding='utf-8') as vocFile:
    vocabulary = build_vocab_from_iterator(vocFile.read().split(", "),
                                           specials=[SOS_TOKEN, EOS_TOKEN, PADDING_TOKEN], special_first=False)

NUMBER_IPA_CHARS = len(vocabulary)-3  # |Σ| = len(vocab)-3


def keepWordsInVoc(words: list[str], ipaVocab: str, nbSamples: int, filename: Optional[str]) -> str:
    """
    Keeps only `nbSamples` words containing only IPA characters defined by `ipaVocab` from word list `words`.

    Args:
        words: list of words to validate.
        ipaVocab: string of IPA characters representing the entire vocabulary. Regex search patterns.
        nbSamples: number of samples to keep from the word list.
        filename (Optional): name of the output file.

    Returns:
        str: a string of the kept text. If a file name is chosen, then this string will be saved in this text file.
    """
    random.seed(42)  # Fix the randomness
    ipaRegex = re.compile(r"^[{0}]+$".format(ipaVocab))

    validSampledWords = random.sample(
        [word for word in words if ipaRegex.match(word)], nbSamples)

    if filename:
        with open(f"{filename}.txt", "w", encoding='utf-8') as f:
            f.write(" ".join(validSampledWords))

    return " ".join(validSampledWords)


def wordsToOneHots(words: list[str], device: Device, inventory: Vocab = vocabulary) -> Tensor:
    """
    Args:
        words: list of words to convert.
        device: device for the returned tensor.
        inventory (Optional): vocabulary (object) for mapping characters into integers. Default value: `vocabulary`.

    Returns:
        IntTensor, dim=(L, B): a padded tensor containing the B words of the list as sequences of L tokens in one-hot indices format.

    Examples:
    >>> wordsToOneHots(['notifikar', 'dɔrikʊ']).T
    tensor([[11, 12, 16,  6,  4,  6,  8,  0, 14],
        [ 2, 30, 14,  6,  8, 43, 59, 59, 59]], dtype=torch.int32)
    """
    return pad_sequence([torch.tensor(inventory(list(word)), dtype=torch.int, device=device) for word in words], batch_first=False, padding_value=inventory[PADDING_TOKEN])


def paddedOneHotsToRawSequencesList(batch: InferenceData) -> list[Tensor]:
    """
    Args:
        batch: the first element must be a padded IntTensor of shape (L, B).

    Returns:
        list[Tensor]: a list of unidimensional IntTensors of variable length.
    """
    return unpad_sequence(batch[0][1:], batch[1], batch_first=False)


def oneHotsToWords(batch: Tensor, removeSpecialTokens: bool = False, inventory: Vocab = vocabulary) -> list[str]:
    """
    Args:
        batch (dim=(L, B)): IntTensor/LongTensor representing padded words in sequences of one hot indices.
        removeSpecialTokens (Optional): argument to remove or not the special tokens. Default value: `False`.
        inventory (Optional): vocabulary (object) for mapping indices into characters. Default value: `vocabulary`.

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
    specialTokensPattern = re.compile('|'.join(
        ['(' + specialToken + ')' for specialToken in (PADDING_TOKEN, SOS_TOKEN, EOS_TOKEN)]))

    wordList = ["".join(inventory.lookup_tokens(wordInLst))
                for wordInLst in batch.T.tolist()]

    if not removeSpecialTokens:
        return wordList

    return specialTokensPattern.sub("", " ".join(wordList)).split(" ")


def __computeInferenceData(wordsTensor: Tensor, vocab: Vocab = vocabulary) -> InferenceData:
    """
    Computes data for the inference from an IntTensor containing words in one-hot indices format.
    The CPU IntTensor is reduced, then the lengths of the sequences are computed and finally
    the one-hot vector encoding is performed.

    Args:
        wordsTensor (dim=(ArbitrarySequenceLength, *)) : IntTensor with the encoded words.
        vocab (Optional): vocabulary containing the mapping between characters and indices. Default value: `vocabulary`.

    Returns:
        InferenceData: a computed data for inference. 
    """
    rawShape = wordsTensor.size()
    device = wordsTensor.device

    withBoundariesTensor = torch.cat((
        torch.full((1, *rawShape[1:]), vocab[SOS_TOKEN],
                   device=device, dtype=torch.int32),
        wordsTensor,
        torch.full((1, *rawShape[1:]), vocab[PADDING_TOKEN],
                   device=device, dtype=torch.int32)
    ))

    mask = torch.logical_xor(
        withBoundariesTensor[:-1] != vocab[PADDING_TOKEN], withBoundariesTensor[1:] != vocab[PADDING_TOKEN])

    withBoundariesTensor[1:] = torch.where(
        mask, vocab[EOS_TOKEN], withBoundariesTensor[1:])

    lengths = torch.argmax(mask.to(torch.uint8), dim=0) + 2
    maxLength = int(torch.max(lengths).item())

    return (withBoundariesTensor[:maxLength], lengths.cpu(), maxLength)


def computeInferenceData_Samples(wordsTensor: Tensor, vocab: Vocab = vocabulary) -> InferenceData_Samples:
    """
    Computes samples' input data for the ReconstructionModel from an IntTensor containing words in one-hot indices format.
    The CPU IntTensor is reduced, then the lengths of the sequences are computed and finally the encoding is performed.
    The one-hot indices IntTensor is of shape (|x|+2, c, b) and the lengths cpu Tensor is of shape (c, b).

    Args:
        wordsTensor: IntTensor of shape (|x|, c) or (|x|, c, b).
        vocab (Optional): vocabulary containing the mapping between characters and indices. Default value: `vocabulary`.

    Returns:
        InferenceData_Samples: computed samples for inference.
    """
    dimensions_number = wordsTensor.ndim

    if dimensions_number == 2:
        wordsTensor = wordsTensor.unsqueeze(-1)
    elif dimensions_number != 3:
        raise Exception("Your tensor with samples' tokens must be of shape (|x|, c) or (|x|, c, b)")

    return __computeInferenceData(wordsTensor, vocab)


def computeInferenceData_Cognates(wordsTensors: dict[ModernLanguages, Tensor], vocab: Vocab = vocabulary) -> dict[ModernLanguages, InferenceData_Cognates]:
    """
    Computes cognates' input data for the ReconstructionModel from IntTensors containing words in one-hot indices format.
    The CPU IntTensors are reduced, then the lengths of the sequences are computed and finally the encoding is performed.
    The EOS token is removed and so the lengths Tensor equals (|y|+1). See the `InferenceData_Cognates` type's documentation for more details.
    
    Args:
        words_intTensor: a dictionnary of IntTensor of shape (|y_l|, c).
        vocab (Optional): vocabulary containing the mapping between characters and indices. Default value: `vocabulary`.

    Returns:
        InferenceData_Samples: computed cognates for inference.
    """
    d:dict[ModernLanguages, InferenceData_Cognates] = {}

    for lang in wordsTensors:
        targets, rawLengths, maxLength = __computeInferenceData(wordsTensors[lang], vocab)
        d[lang] = (targets.where(targets != vocab[EOS_TOKEN], vocab[PADDING_TOKEN])[:-1],
            rawLengths-1, maxLength-1)

    return d