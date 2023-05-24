from Source.generateProposals import getMinEditPaths, computeProposals, computeMinEditDistanceMatrix
from data.vocab import make_oneHotTensor
import torch

def bigDataTest():
    a,b = ("arɡymntaθjon", "ˌɐɾəɡumˌeɪŋtɐsˈɐ̃ʊ̃")
    print(computeMinEditDistanceMatrix(a, b)[len(a), len(b)])
    tree = getMinEditPaths(a, b)
    tree.displayGraph("errorGraph")
    proposals = computeProposals(a, b)
    #tensor = make_oneHotTensor(proposals, True)

samples = [("absɛns", "assɛnte"), ("abɛrɾasɔ", "aberɾatsiˈone"), ("lɛɡˈatɪɔ","leɡasjˈɔ̃"), ("paɾaθentɛzɪs", "paɾatʃentˈezɪ"), ("partiθipʊ", "partitʃˈipio")]

def variousDataTest(save:bool = False):    
    goodProposals = torch.load('./Tests/proposalsTensors.pt')
    proposals = []
    for i in range(len(samples)):
        a, b = samples[i]
        p = computeProposals(a, b)
        if save:
            proposals.append(p)
        else:
            assert(torch.all(goodProposals[i]==p))
    if save:
        torch.save(proposals, f'./Tests/proposalsTensors.pt')

def drawGraphs():
    for i in range(len(samples)):
        a,b = samples[i]
        editGraph = getMinEditPaths(a, b)
        editGraph.displayGraph(str(i))