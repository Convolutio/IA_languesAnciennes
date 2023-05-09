from Source.generateProposals import getMinEditPaths, computeProposals, editProtoForm, computeMinEditDistanceMatrix
from data.vocab import make_tensor
import numpy as np
import multiprocessing as mp

def bigDataTest():
    a,b = "arɡymntaθjon", "ˌɐɾəɡumˌeɪŋtɐsˈɐ̃ʊ̃"
    print(computeMinEditDistanceMatrix(a, b)[len(a), len(b)])
    tree = getMinEditPaths(a, b)
    tree.displayGraph("errorGraph")
    proposals = computeProposals(a, b)
    tensor = make_tensor(proposals, True)

def variousDataTest():
    samples = [("absɛns", "assɛnte"), ("abɛrɾasɔ", "aberɾatsiˈone"), ("lɛɡˈatɪɔ","leɡasjˈɔ̃")]
    for i in range(len(samples)):
        a, b = samples[i]
        p = computeProposals(a, b)
        # if i==0:
        #     print(p)
        