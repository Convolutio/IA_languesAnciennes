from torch import zeros, cuda
from typing import Literal

device = 'cuda' if cuda.is_available() else 'cpu'
class ProbCache:
    """
    dim = (|x|+2, |y|+2, B, 1)
    Usefull cached data in the spaces below:
        * sub : (|x|+1, |y|)
        * ins : (|x|+1, |y|)
        * dlt : (|x|+1, |y|+1)
        * end : (|x|+1, |y|+1)
    Padding value : 0
    """
    def __init__(self, maxSourceLength:int, maxTargetLength:int, batch_size:tuple[int, Literal[1]]):
        """
        Arguments:
            - maxSourceLength : the maximum source sequence length (with boundaries)
            - maxTargetLength : the maximum target sequence length (with boundaries)
        """
        self.sub = zeros((maxSourceLength, maxTargetLength, *batch_size), device=device)
        self.ins = zeros((maxSourceLength, maxTargetLength, *batch_size), device=device)
        self.dlt = zeros((maxSourceLength, maxTargetLength, *batch_size), device=device)
        self.end = zeros((maxSourceLength, maxTargetLength, *batch_size), device=device)
