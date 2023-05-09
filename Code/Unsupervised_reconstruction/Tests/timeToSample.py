from Source.generateProposals import computeProposals, computeMinEditDistanceMatrix
from data.getDataset import getCognatesSet, getIteration
from data.vocab import make_tensor

def generateProposalsOverDataset():
    cognates = getCognatesSet()
    reconstructions = getIteration(4)
    numberOfCognatePairs = len(cognates['french'])
    for i in range(numberOfCognatePairs):
        x = reconstructions[i]
        for language in ('spanish', 'portuguese', 'italian', 'romanian', 'french'):
            y = cognates[language][i]
            editDistance = computeMinEditDistanceMatrix(x, y)[len(x), len(y)]
            if editDistance < 16:
                print(f'/{x}/ to /{y}/ -{editDistance}- (<{language}>)\nIteration {str(i)}/{numberOfCognatePairs}', end='\r')
                p = computeProposals(x,cognates[language][i])
                t = make_tensor(p, True)

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
