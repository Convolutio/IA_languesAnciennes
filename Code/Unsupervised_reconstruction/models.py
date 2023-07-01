from numpy import zeros

class ProbCache:
    """
    dim = (|x|+2, |y|+2, B)
    Usefull cached data in the spaces below:
        * sub : (|x|+1, |y|)
        * ins : (|x|+1, |y|)
        * dlt : (|x|+1, |y|+1)
        * end : (|x|+1, |y|+1)
    Padding value : 0
    """
    def __init__(self, maxSourceLength:int, maxTargetLength:int, batch_size:int):
        self.sub = zeros((maxSourceLength, maxTargetLength, batch_size))
        self.ins = zeros((maxSourceLength, maxTargetLength, batch_size))
        self.dlt = zeros((maxSourceLength, maxTargetLength, batch_size))
        self.end = zeros((maxSourceLength, maxTargetLength, batch_size))
