from enum import Enum
from typing import Union, TypedDict, Literal
import numpy.typing as npt
import numpy as np

ModernLanguages = Literal['french', 'spanish', 'italian', 'portuguese', 'romanian']
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