from typing import Literal, Union, NamedTuple
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


# --- GLOBAL CONSTANTS ---
PADDING_TOKEN = '-'
SOS_TOKEN = '('
EOS_TOKEN = ')'


# --- GENERAL ALIASES ---
# Maybe remove ?
rawCognate = str
rawSample = str
rawTarget = str

tensorCognate = Tensor
tensorSample = Tensor
tensorTarget = Tensor


# --- EDIT ALIASES ---
EditsCombination = Tensor
Edit = tuple[Literal[0, 1, 2], int, int]
"""
[op, i, j]

substitution : [0, i, j]\\
deletion : [1, i, j]\\
insertion : [2, i, j]

$j$ is the index of the character of y. It enables to differentiate an operation which
takes place at different positions in the edit distance matrix, but in which i and i' have
the same value and so do y[j] and y[j'], and the operation is identical (it is a case which
has already been met). It also enables to insert characters in the same i position of x
in the good order, whatever the order with which the insertions have been applied.  
The positionnal index $i$ is between -1 and |x|-1 included, $i=-1$ meaning an insertion on
the left of the word.
"""

# --- MODERN LANGUAGES ---
ModernLanguages = Literal['french', 'spanish',
                          'italian', 'portuguese', 'romanian']
MODERN_LANGUAGES: tuple[ModernLanguages, ...] = (
    'french', 'spanish', 'italian', 'portuguese', 'romanian')

# --- OPERATIONS ---
Operations = Literal['sub', 'dlt', 'ins', 'end']
OPERATIONS: tuple[Operations, ...] = ("sub", "dlt", "ins", "end")


# --- Inference Data Classes ---
class InferenceData(NamedTuple):
    # InferenceData = tuple[Tensor, Tensor, int]
    """
    The data in a language for inferring into an edit model with the dynamic program.

    This tuple contains :
        * the tensor with the batch's words with their boundaries, encoded into one-hot indexes sequence.
        (shape = (N+2, C, B) or (N+2, C)), with C the number of cognate pairs, B the number of samples linked to each cognate pair (1 if it is a cognate, else it is a number of proposals linked to each cognate), N the variable raw sequences lengths).
        See data.vocab.py for the meaning of each one-hot indexes. 
        * the CPU IntTensor with the lengths of the sequences (with their boundaries). (shape = (C, B) or (C))
        * The max length of a sequence in the batch, for avoiding computing again this information.
    """
    tensor: Tensor
    lengths: Tensor
    maxLength: int


class InferenceData_Samples(NamedTuple):
    # InferenceData_Samples = tuple[Tensor, Tensor, int]
    # Do not derived this dataclass from InferenceData
    """
    This is the samples' expected input data type for the ReconstructionModel
    Tuple arguments:
        * IntTensor with the samples with one-hot indexes format. They are sequences of tokens with IPA chararacters and the boundary tokens. shape = (|x|+2, C, B)
        * CPU IntTensor with the lengths of sequences (with the boundary tokens, so |x|+2). shape = (C, B)
        * int with the maximum one
    """
    samples: Tensor
    lengths: Tensor
    maxLength: int


class InferenceData_SamplesEmbeddings(NamedTuple):
    # InferenceData_SamplesEmbeddings = tuple[PackedSequence, Tensor, int]
    # Do not derived this dataclass from InferenceData
    """
    It is expected the source input data to be passed in an EditModel with the PackingEmbedding conversion which has already been externally applied.

    Tuple arguments:
        * PackedSequence of the samples' embeddings, which are sequences of tokens with the boundaries. shape = (|x|+2, C*B, input_dim)
        * CPU IntTensor with the lengths of sequences (with boundaries, so |x|+2). shape = (C, B)
        * int with the maximum one
    """
    sampleEmbs: PackedSequence
    lengths: Tensor
    maxLength: int


class InferenceData_Cognates(NamedTuple):
    # InferenceData_Cognates = tuple[Tensor, Tensor, int]
    # Do not derived this dataclass from InferenceData
    """
    Tuple arguments:
        * IntTensor with the modern forms with one-hot indexes format. They are sequences of tokens with IPA chararacters and only the opening boundary. shape = (|y|+1, C)
        * CPU IntTensor with the lengths of sequences (with the opening boundary, so |y|+1). shape = (C)
        * int with the maximum one
    """
    cognates: Tensor
    lengths: Tensor
    maxLength: int
