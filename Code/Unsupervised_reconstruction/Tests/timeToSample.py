from Source.generateProposals import computeProposals
from data.getDataset import getCognatesSet, getIteration

cognates = getCognatesSet()
reconstructions = getIteration(4)
numberOfCognatePairs = len(cognates['french'])
for i in range(numberOfCognatePairs):
    for language in ('spanish', 'portuguese', 'italian', 'romanian'):
        computeProposals(reconstructions[i],cognates[language][i])
        print(f'Iteration {str(i)}/{numberOfCognatePairs}: (on <{language}>)', end='\r')