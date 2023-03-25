from enum import Enum

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

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

Edition = tuple[op, sub_char|ins_char, Form, int, Form]
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
        self.output_dim = output_dim #must be equal to |Σ|+1 for the <end> and <del> characters
        
        self.encoder_prior = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        #les dimensions des couches cachées sont multipliées par 2 pour être égale à
        #celle de la couche cachée de l'encodeur bidirectionnel (qui subit une concaténation)
        self.encoder_modern = nn.LSTM(input_dim, hidden_dim*2)

        self.sub_head = nn.Sequential(
            nn.Linear(hidden_dim*2, output_dim),
            nn.Softmax()
            )
        self.ins_head = nn.Sequential(
            nn.Linear(hidden_dim*2, output_dim),
            nn.Softmax()
            )

    def __encode_context(self, x:Form, i:int, y0:Form):
        encoder_prior__output, _ = self.encoder_prior(x) # cette inférence s'exécute plusieurs fois inutilement !!
        h_xi = encoder_prior__output[i, :]
        encoder_modern__output, _ = self.encoder_modern(y0)
        g_y0 = encoder_modern__output[-1, :]
        return h_xi + g_y0
    
    def q_sub(self, x:Form, i:int, y0:Form):
        """
        Forwards inference in q_sub model.
        Returns a tensor (dimension : |Σ*|) representing a probability
        distribution over Σ* that each token Σ*[j] is added to y0 in 
        replacing the x[i] token.
        """
        context = self.__encode_context(x, i, y0)
        return self.sub_head(context)
    
    def q_ins(self, x:Form, i:int, y0:Form)->torch.Tensor:
        """
        Forwards inference in q_ins model.
        Returns a tensor (dimension : |Σ*|) representing a probability
        distribution over Σ* that each token Σ*[j] is inserted after y0.
        """
        context = self.__encode_context(x, i, y0)
        return self.sub_ins(context)
    
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
            omega:IPA_char = SIGMA_SUB[torch.argmax(self.q_sub(x, i, y0))]
            delta.append(
                (op.sub, omega, x.copy(), i, y0.copy())
                )
            if omega!='<del>':
                canInsert:bool = True
                while canInsert:
                    y0.append(omega)
                    omega = SIGMA_INS[torch.argmax(self.q_ins(x, i, y0))]
                    delta.append(
                        (op.ins, omega, x.copy(), i, y0.copy())
                    )
                    canInsert = omega!="<end>"
        return (y0.copy(), delta)
    