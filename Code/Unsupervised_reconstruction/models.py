from numpy import zeros

class ProbCache:
    def __init__(self, maxSourceLength:int, maxTargetLength:int, batch_size:int):
        self.sub = zeros((maxSourceLength, maxTargetLength, batch_size))
        self.ins = zeros((maxSourceLength, maxTargetLength, batch_size))
        self.dlt = zeros((maxSourceLength, maxTargetLength, batch_size))
        self.end = zeros((maxSourceLength, maxTargetLength, batch_size))
