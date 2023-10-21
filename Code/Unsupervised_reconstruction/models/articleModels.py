from enum import Enum
from typing import Literal
from torch import Tensor

ModernLanguages = Literal['french', 'spanish', 'italian', 'portuguese', 'romanian']
MODERN_LANGUAGES: tuple[ModernLanguages, ...] = ('french', 'spanish', 'italian', 'portuguese', 'romanian')
Operations = Literal['sub', 'dlt', 'ins', 'end']
OPERATIONS:tuple[Operations,...] = ("sub", "dlt", "ins", "end")

CognatesSet_str = dict[ModernLanguages, list[str]]
CognatesSet_oneHotIdxs = dict[ModernLanguages, list[Tensor]]

Form = str
FormsSet = list[str] # a list of forms