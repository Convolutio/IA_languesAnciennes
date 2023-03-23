from enum import Enum

import torch
import torch.nn as nn

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

Form = list[IPA_char]

Edition = tuple[op, IPA_char, Form, int, Form]

class EditModel(nn.Module):
    """
    This class gather the edit models
    and edit function specific to the 
    modern Language
    """
    def __init__(self, languages:Languages, emb_dim: int, hidden_dim: int, output_dim: int):
        super(EditModel, self).__init__()
        self.languages = languages

        self.input_encoder = nn.LSTM(input_size = emb_dim, hidden_size = hidden_dim, bidirectional = True)
        self.output_encoder = nn.LSTM(input_size = emb_dim, hidden_size = hidden_dim)

    def forward(self):
        pass

    def edit(self, x: Form) -> tuple[Form, list[Edition]]:
        """
        Args:
            x (Sequence[str]): a list of IPA tokens representing the ancestral form
        
        Returns
            y : a list of tokens representing a modern form 
            delta: lists of edits
        """
        y_temp: Form = [] # it is represented as y' in the figure 2 of the paper
        delta: list[Edition] = []
        for i in range(len(x)):
            omega:IPA_char = self.q_sub.sample(x, i, y_temp)
            delta.append(
                (op.sub, omega, x.copy(), i, y_temp.copy())
                )
            if omega!='<del>':
                canInsert:bool = True
                while canInsert:
                    y_temp.append(omega)
                    omega = self.q_ins.sample(x, i, y_temp)
                    delta.append(
                        (op.ins, omega, x.copy(), i, y_temp.copy())
                    )
                    canInsert = omega!="<end>"
        return (y_temp.copy(), delta)
    
    def training(self):
        pass

    def evaluation(self):
        pass