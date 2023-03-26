from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim  

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
    
    def training(self, train_loader, test_loader, epochs=5, learning_rate=0.01):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # for epoch in range(epochs):
        #     running_loss = 0.0
        #     for inputs in train_loader:
        #         optimizer.zero_grad()         
        #         outputs = self(inputs)
        #         loss = outputs.loss
        #         loss.backward()
        #         optimizer.step()
        #         running_loss += loss.item()
                
        #     print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
        #     self.evaluation(test_loader)

    def evaluation(self, test_loader):
        """
        Evaluate the performance of our model by computing the average edit distance between its outputs
        and gold Latin reconstructions.
        """
        correct = 0
        total = 0
        # with torch.no_grad():
        #     for data in test_loader:
        #         outputs = self(data)
        #         _, predicted = torch.max(outputs.data, 1)
        #         # edit distance
        
        # print(f'Test Accuracy: {100 * correct / total}%')

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
    