import numpy as np

INFTY_NEG = -1e9
def computeLaw(probs:np.ndarray)->np.ndarray:
    """
    Arguments:
        probs (list[float]): the list of logarithmic unnormalized probs
    Returns a logarithmic probabilty distribution over all the probs
    """
    totalProb = INFTY_NEG
    for prob in probs:
        totalProb = np.logaddexp(totalProb, prob)
    return np.array([prob-totalProb for prob in probs], dtype=float)

M = 10**4 # normally ok
def metropolisHasting(law:np.ndarray)->int:
    """
    Arguments:
        law (list[float]) : a probability distribution
    Sample the index of a proposal randomly from the probability
    distribution.
    """
    N = len(law)
    i:int = 0
    for _ in range(M):
        j = np.random.randint(0, N-1)
        if j >= i:
            j = j+1
        acceptation = law[j] - law[i]
        u = np.log(np.random.uniform())
        if u <= acceptation:
            i = j
    return i

