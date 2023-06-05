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

def optimizedMH_Test():
    """
    Run MH a lot of times and compute the frequencies of indexes selections.
    """
    M = 10**5
    N = len(law)
    t_law = torch.tensor(law)
    i = torch.zeros(REPETITIONS, dtype=torch.int32)
    for _ in range(M):
        j = torch.randint(0, N-1, (REPETITIONS,), dtype=torch.int32)
        j = torch.where(j>=i, j+1, j)
        acceptation = torch.index_select(t_law, 0, j) - torch.index_select(t_law, 0, i)
        u = torch.log(torch.rand(REPETITIONS))
        i = torch.where(u<=acceptation, j, i)
    return torch.exp(torch.log(torch.bincount(i))-torch.log(REPETITIONS))

print("Frequencies:", optimizedMH_Test())