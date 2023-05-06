from Source.generateProposals import getMinEditPaths, computeProposals, editProtoForm, computeMinEditDistanceMatrix

a,b = "arɡymntaθjon", "ˌɐɾəɡumˌeɪŋtɐsˈɐ̃ʊ̃"
print(computeMinEditDistanceMatrix(a, b)[len(a), len(b)])
tree = getMinEditPaths(a, b)
tree.displayGraph("errorGraph", "random")
computeProposals(a, b)
# samples = [("abɛrɾasɔ", "aberɾatsiˈone"), ("absɛns", "assɛnte"), ("lɛɡˈatɪɔ","leɡasjˈɔ̃")]
# for i in range(len(samples)):
#     a, b = samples[i]
#     editsGraph = getMinEditPaths(a, b)
#     editsGraph.displayGraph(filename=str(i), comment=f"Initial edits graph for \"{a}\" to \"{b}\"")
#     if i==1:
#         print(computeProposals(a, b))
#     else:
#         computeProposals(a,b)