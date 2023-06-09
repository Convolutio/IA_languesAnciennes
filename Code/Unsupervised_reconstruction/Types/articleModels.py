from enum import Enum
from typing import Union, Literal
import numpy.typing as npt
import numpy as np

ModernLanguages = Literal['french', 'spanish', 'italian', 'portuguese', 'romanian']
class op(Enum):
    sub='sub'
    ins='ins'

CognatesSet = dict[ModernLanguages, list[str]]

Form = str
FormsSet = list[str] # a list of forms, 
OneHotsSet = npt.NDArray[np.bool_]
Edition = tuple[op, Union[str,str], Form, int, Form]