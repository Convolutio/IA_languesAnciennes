from Source.generateProposals import computeProposals
from data.getDataset import getCognatesSet, getIteration

cognates = getCognatesSet()
reconstructions = getIteration(4)
numberOfCognatePairs = len(cognates['french'])
for i in range(210, numberOfCognatePairs):
    for language in ('spanish', 'portuguese', 'italian', 'romanian'):
        print(f'/{reconstructions[i]}/ to /{cognates[language][i]}/ (<{language}>)\nIteration {str(i)}/{numberOfCognatePairs}', end='\r')
        computeProposals(reconstructions[i],cognates[language][i])