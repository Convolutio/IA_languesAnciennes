from enum import Enum
from typing import Union, TypedDict
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

class CognatesSet(TypedDict):
    french:list[str]
    spanish:list[str]
    italian:list[str]
    portuguese:list[str]
    romanian:list[str]

Form = str
FormsSet = list[str] # a list of forms, 
OneHotsSet = npt.NDArray[np.bool_]
Edition = tuple[op, Union[str,str], Form, int, Form]