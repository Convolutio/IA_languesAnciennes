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
    def __init__(self, languages:Languages, input_dim: int, hidden_dim: int, output_dim: int):
        super(EditModel, self).__init__()
        self.languages = languages

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.encoder = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim)

        self.sub_head = nn.Linear(hidden_dim, output_dim)
        self.ins_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, y0):
        encoder_output, _ = self.encoder(x)
        h_x = encoder_output[:, -1, :]
        
        decoder_output, _ = self.decoder(y0)
        g_y0 = decoder_output[-1, :]
        
        context = h_x + g_y0
        
        qsub = self.sub_head(context)
        qins = self.ins_head(context)
        
        return qsub, qins
    
    def training(self):
        pass

    def evaluation(self):
        pass

    def edit(self, x: Form) -> tuple[Form, list[Edition]]:
        """
        Args:
            x (Sequence[str]): a list of IPA tokens representing the ancestral form
        
        Returns
            y : a list of tokens representing a modern form 
            delta: lists of edits
        """
        y0: Form = [] # it is represented as y' in the figure 2 of the paper
        delta: list[Edition] = []
        for i in range(len(x)):
            omega:IPA_char = self.q_sub.sample(x, i, y0)
            delta.append(
                (op.sub, omega, x.copy(), i, y0.copy())
                )
            if omega!='<del>':
                canInsert:bool = True
                while canInsert:
                    y0.append(omega)
                    omega = self.q_ins.sample(x, i, y0)
                    delta.append(
                        (op.ins, omega, x.copy(), i, y0.copy())
                    )
                    canInsert = omega!="<end>"
        return (y0.copy(), delta)
    