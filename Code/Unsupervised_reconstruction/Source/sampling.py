import numpy as np

INFTY_NEG = -1e9


def computeLaw(probs: np.ndarray) -> np.ndarray:
    """
    Returns a logarithmic probabilty distribution over all the probs.

    Arguments:
        probs (list[float]): the list of logarithmic unnormalized probs
    """
    totalProb = INFTY_NEG

    for prob in probs:
        totalProb = np.logaddexp(totalProb, prob)

    return np.array([prob-totalProb for prob in probs], dtype=float)


def metropolisHasting(law: np.ndarray, iteration: int = 10**4) -> int:
    """
    Sample the index of a proposal randomly from the probability
    distribution.

    Arguments:
        law (list[float]) : a probability distribution in logarithm
    """
    N = len(law)
    i: int = 0

    for _ in range(iteration):
        j = np.random.randint(0, N-1)
        
        if j >= i: j = j+1

        acceptation = law[j] - law[i]
        u = np.log(np.random.uniform())

        if u <= acceptation: i = j

    return i
