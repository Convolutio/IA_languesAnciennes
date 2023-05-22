from Source.generateProposals import getMinEditPaths, computeProposals, computeMinEditDistanceMatrix
from data.vocab import make_oneHotTensor
import numpy as np
import multiprocessing as mp

def bigDataTest():
    a,b = ("arɡymntaθjon", "ˌɐɾəɡumˌeɪŋtɐsˈɐ̃ʊ̃")
    print(computeMinEditDistanceMatrix(a, b)[len(a), len(b)])
    tree = getMinEditPaths(a, b)
    tree.displayGraph("errorGraph")
    proposals = computeProposals(a, b)
    #tensor = make_oneHotTensor(proposals, True)

def variousDataTest():
    samples = [("absɛns", "assɛnte"), ("abɛrɾasɔ", "aberɾatsiˈone"), ("lɛɡˈatɪɔ","leɡasjˈɔ̃")]
    for i in range(len(samples)):
        a, b = samples[i]
        p = computeProposals(a, b)
        # if i==0:
        #     print(p)

def drawGraphs():
    samples = [("absɛns", "assɛnte"), ("abɛrɾasɔ", "aberɾatsiˈone"), ("lɛɡˈatɪɔ","leɡasjˈɔ̃"),
               ("arɡymntaθjon", "ˌɐɾəɡumˌeɪŋtɐsˈɐ̃ʊ̃")]
    for i in range(len(samples)):
        a,b = samples[i]
        editGraph = getMinEditPaths(a, b)
        editGraph.displayGraph(str(i))