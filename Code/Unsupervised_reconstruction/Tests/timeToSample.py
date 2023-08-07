from source.generateProposals import computeProposals, computeMinEditDistanceMatrix
from data.getDataset import getCognatesSet, getIteration
from data.vocab import make_oneHotTensor


def generateProposalsOverDataset():
    p = list()
    cognates = getCognatesSet()
    reconstructions = getIteration(4)
    numberOfCognatePairs = len(cognates['french'])
    languages = ('spanish', 'portuguese', 'italian', 'romanian', 'french')

    for i in range(numberOfCognatePairs):
        x = reconstructions[i]
        Y = [cognates[language][i] for language in languages]
        print(f'/{x}/ to ({Y}) \nCognate Set {str(i)}/{numberOfCognatePairs}'+''*10, end='\r')
        p.append(computeProposals(x, Y))

def editDistancesInDataset():
    cognates = getCognatesSet()
    reconstructions = getIteration(4)
    numberOfCognatePairs = len(cognates['french'])
    editDistances = {i: 0 for i in range(17)}

    for i in range(numberOfCognatePairs):
        x = reconstructions[i]
        for language in ('spanish', 'portuguese', 'italian', 'romanian', 'french'):
            y = cognates[language][i]
            d = computeMinEditDistanceMatrix(x, y)[len(x), len(y)]
            editDistances[d] += 1

    print(editDistances)
