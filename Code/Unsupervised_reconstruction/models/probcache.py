from torch import zeros, cuda, Tensor, where
from typing import Literal
from source.utils import computePaddingMask
from models.articleModels import Operations, OPERATIONS
from models.models import SourceInferenceData, TargetInferenceData

device = 'cuda' if cuda.is_available() else 'cpu'


class ProbCache:
    """
    dim = (|x|+2, |y|+2, B, 1)
    Call `toTargetsProbs` method once the cached probs are all set up.
    Usefull cached data in the spaces below:
        * sub : (|x|+1, |y|)
        * ins : (|x|+1, |y|)
        * dlt : (|x|+1, |y|+1)
        * end : (|x|+1, |y|+1)
    Padding value : 0
    """
    def __init__(self, sourcesData:SourceInferenceData, targetsData:TargetInferenceData, batch_size:tuple[int, Literal[1]]):
        """
        Arguments:
            - sourcesData : needed to get length data
            - maxTargetLength : needed to get length data
        """
        maxSourceLength = sourcesData[2]
        maxTargetLength = targetsData[2] + 1
        self.sourceLengthData = (sourcesData[1]-1, sourcesData[2]-1)
        self.targetLengthData = (targetsData[1].unsqueeze(-1), targetsData[2])
        self.sub = zeros((maxSourceLength, maxTargetLength, *batch_size), device=device).detach()
        self.ins = zeros((maxSourceLength, maxTargetLength, *batch_size), device=device).detach()
        self.dlt = zeros((maxSourceLength, maxTargetLength, *batch_size), device=device).detach()
        self.end = zeros((maxSourceLength, maxTargetLength, *batch_size), device=device).detach()

    def toTargetsProbs(self) -> list[dict[Operations, Tensor]]:
        """
        Return, for each sample, its target edit probabilities. The undefined probabilities are neutralized to 0 in log space.
        Tensors shape = (|x|+1, |y|+1, 1, 1)
        """
        paddingMask = computePaddingMask(self.sourceLengthData, self.targetLengthData)
        for op in OPERATIONS:
            setattr(self, op, where(paddingMask, getattr(self, op)[:-1, :-1], 0))
        
        d: dict[Operations, list[Tensor]] = {op:getattr(self, op).split(1,2) for op in OPERATIONS}
        return [{op:d[op][i] for op in d} for i in range(len(d['dlt']))]