from Source.sampling import computeLaw
import numpy as np
import torch
import math

REPETITIONS = 10**5
THREADS_NUMBER = 20

unnormalizedProbs = np.array([0.1, 0.15, 0.08, 0.11, 0.144, 0.078])
logProbs = np.log(unnormalizedProbs, dtype=float)
law = computeLaw(logProbs)
print("Computed law:", np.exp(law))
law = logProbs

def optimizedMH_Test(samplingIterations:int):
    """
    Run MH a lot of times and compute the frequencies of indexes selections.
    """
    MH_iterations = 10**5
    N = len(law)
    t_law = torch.tensor(law)
    i = torch.zeros(samplingIterations, dtype=torch.int32)
    for _ in range(MH_iterations):
        j = torch.randint(0, N-1, (samplingIterations,), dtype=torch.int32)
        j = torch.where(j>=i, j+1, j)
        acceptation = torch.index_select(t_law, 0, j) - torch.index_select(t_law, 0, i)
        u = torch.log(torch.rand(samplingIterations))
        i = torch.where(u<=acceptation, j, i)
    return torch.exp(torch.log(torch.bincount(i))-math.log(samplingIterations))

print("Frequencies:", optimizedMH_Test(REPETITIONS))