from source.generateProposals import getMinEditPaths, computeProposals, computeMinEditDistanceMatrix
import torch

def bigDataTest():
    a,b = ("arɡymntaθjon", "ˌɐɾəɡumˌeɪŋtɐsˈɐ̃ʊ̃")
    print(computeMinEditDistanceMatrix(a, b)[len(a), len(b)])
    tree = getMinEditPaths(a, b)
    tree.displayGraph("errorGraph")
    proposals = computeProposals(a, b)
    #tensor = make_oneHotTensor(proposals, True)

samples = [("absɛns", "assɛnte"), ("abɛrɾasɔ", "aberɾatsiˈone"), ("lɛɡˈatɪɔ","leɡasjˈɔ̃"), ("paɾaθentɛzɪs", "paɾatʃentˈezɪ"), ("partiθipʊ", "partitʃˈipio"), ("arɡymntaθjon", "ˌɐɾəɡumˌeɪŋtɐsˈɐ̃ʊ̃")]

def variousDataTest(save:bool = False):    
    goodProposals = torch.load('./Tests/proposalsTensors.pt')
    proposals = []
    for i in range(len(samples)):
        a, b = samples[i]
        p = computeProposals(a, b)
        if save:
            proposals.append(p)
        else:
            try:
                assert(torch.all(goodProposals[i]==p))
            except:
                editGraph = getMinEditPaths(a,b)
                editGraph.displayGraph('errorGraph')
                msg = f'Good number of proposals: {goodProposals[i].shape[0]}'
                msg += f'\nCurrent number of proposals: {p.shape[0]}'
                raise Exception(msg)
    if save:
        torch.save(proposals, f'./Tests/proposalsTensors.pt')

def drawGraphs():
    for i in range(len(samples)):
        a,b = samples[i]
        editGraph = getMinEditPaths(a, b)
        editGraph.displayGraph(str(i))