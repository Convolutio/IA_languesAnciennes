from enum import Enum
from typing import Union

class Languages(Enum):
    FRENCH="french"
    SPANISH="spanish"
    ITALIAN="italian"
    PORTUGUESE="portuguese"
    LATIN="latin"

class op(Enum):
    sub='sub'
    ins='ins'

IPA_char = str
sub_char = str
ins_char = str
SIGMA:list[IPA_char] = [] #list of IPA tokens
SIGMA_SUB:list[sub_char] = SIGMA + ["<del>"]
SIGMA_INS:list[ins_char] = SIGMA + ["<end>"]


Form = list[IPA_char]

Edition = tuple[op, Union[sub_char,ins_char], Form, int, Form]