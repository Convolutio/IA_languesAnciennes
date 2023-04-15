from enum import Enum
from typing import Union
import numpy.typing as npt
import numpy as np

class Languages(Enum):
    FRENCH="french"
    SPANISH="spanish"
    ITALIAN="italian"
    PORTUGUESE="portuguese"
    LATIN="latin"

class op(Enum):
    sub='sub'
    ins='ins'


IPA_char = np.str_
sub_char = np.str_
ins_char = np.str_
SIGMA:list[IPA_char] = [] #list of IPA tokens
SIGMA_SUB:list[sub_char] = SIGMA + [np.str_("<del>")]
SIGMA_INS:list[ins_char] = SIGMA + [np.str_("<end>")]


Form = list[IPA_char]
FormsSet = npt.NDArray[np.str_] # a list of forms, with potentially two dimensions if there is a batch size 
Edition = tuple[op, Union[sub_char,ins_char], Form, int, Form]