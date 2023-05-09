from Source.generateProposals import computeProposals, computeMinEditDistanceMatrix
from data.getDataset import getCognatesSet, getIteration

def generateProposalsOverDataset():
    cognates = getCognatesSet()
    reconstructions = getIteration(4)
    numberOfCognatePairs = len(cognates['french'])
    editDistances = {i:0 for i in range(17)}
    for i in range(numberOfCognatePairs):
        x = reconstructions[i]
        for language in ('spanish', 'portuguese', 'italian', 'romanian', 'french'):
            print(f'/{reconstructions[i]}/ to /{cognates[language][i]}/ (<{language}>)\nIteration {str(i)}/{numberOfCognatePairs}', end='\r')
            computeProposals(reconstructions[i],cognates[language][i])
    print(editDistances)

def editDistancesInDataset():
    cognates = getCognatesSet()
    reconstructions = getIteration(4)
    numberOfCognatePairs = len(cognates['french'])
    editDistances = {i:0 for i in range(17)}
    for i in range(numberOfCognatePairs):
        x = reconstructions[i]
        for language in ('spanish', 'portuguese', 'italian', 'romanian', 'french'):
            y = cognates[language][i]
            d = computeMinEditDistanceMatrix(x, y)[len(x), len(y)]
            editDistances[d] += 1
    print(editDistances)
