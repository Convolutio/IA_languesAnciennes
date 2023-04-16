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

Form = str
FormsSet = list[str] # a list of forms, 
OneHotsSet = npt.NDArray[np.bool_]
Edition = tuple[op, Union[str,str], Form, int, Form]