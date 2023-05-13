from Source.sampling import metropolisHasting, computeLaw
import numpy as np
from threading import Thread

def computeFrequencies(law:np.ndarray, iterations:int, effectivesTable:np.ndarray, threadNumber:int):
    for _ in range(iterations):
        n = metropolisHasting(law)
        effectivesTable[threadNumber, n] += 1

REPETITIONS = 10**4
THREADS_NUMBER = 20

unnormalizedProbs = np.array([0.1, 0.15, 0.08, 0.11, 0.144, 0.078])
probs = np.log(unnormalizedProbs, dtype=float)
law = computeLaw(probs)
print("Computed law:", np.exp(law))


print('0%', end='\r')
effectivesByThread = np.zeros((THREADS_NUMBER, law.size)) 
for p in range(100):
    threads = [Thread(target=computeFrequencies, args=(law, 
                REPETITIONS//(100*THREADS_NUMBER), effectivesByThread, i))
                for i in range(THREADS_NUMBER)]
    for tn in range(THREADS_NUMBER):
        threads[tn].start()
    for tn in range(THREADS_NUMBER):
        threads[tn].join()
    print(f'{p+1}%', end='\r')
print("100%")
frequencies = effectivesByThread.sum(0)/REPETITIONS

print("Frequencies:", frequencies)