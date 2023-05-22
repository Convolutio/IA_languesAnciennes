from Types.models import *
from typing import Optional
import graphviz
from collections import deque
from torch import Tensor, uint8, ByteTensor
import torch
from data.vocab import wordToOneHots

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Node:
    def __init__(self, id_:int) -> None:
        self.__id = id_
        self.__parentVertecies:set[int] = set() #ids of vertecies from which the node can come
        self.__childVertecies:set[int] = set() #ids of vertecies directly joinable from the node

    @property
    def id_(self):
        return self.__id
    
    @property
    def childVertecies(self):
        return self.__childVertecies
    
    @property
    def parentVertecies(self):
        return self.__parentVertecies
    
    def addChildVertex(self, vertexId:int):
        self.__childVertecies.add(vertexId)
    
    def removeChildVertex(self, vertexId:int):
        self.__childVertecies.remove(vertexId)
    
    def addParentVertex(self, vertexId:int):
        self.__parentVertecies.add(vertexId)
    
    def removeParentVertex(self, vertexId:int):
        self.__parentVertecies.remove(vertexId)

def addElementToCombination(combi:Tensor, element:Tensor)->Tensor:
    newCombi = torch.empty_like(combi).copy_(combi)
    oldcombinationsNumber = int(combi[0, 0].item())
    newCombi[oldcombinationsNumber+1] = element
    newCombi[0, 0] = oldcombinationsNumber + 1
    return torch.unsqueeze(newCombi, dim=0) # returns a combination line

class EditsGraph:
    """
    This class manages the features of an oriented graph to represent
    the edit paths.
    """
    def __init__(self, x:str, y:str, editDistance:int) -> None:
        self.__editIds:dict[Edit, int] = {}
        self.__edits:list[Edit] = [(0,0,0)]
        self.__insertionInfos:list[list[int]] = [[0,len(y)] for i in range(len(x)+1)]
        """
        The matrix above contains for each i position in x the following information:\\
        [the max number of inserted characters, the minimal j index in y of the inserted characters]
        """
        # The dict below saves the graph's vertecies and their connexions.
        self.__nodes:list[Node] = [Node(0)] # Empty beginning node with 0 index
        self.__editDistance = editDistance
        #For debugging
        self.__x = x
        self.__y = y

    def __addNode(self, edit:Edit):
        newNodeId = len(self.__nodes)
        self.__edits.append(edit)
        self.__nodes.append(Node(newNodeId))
        self.__editIds[edit] = newNodeId
        if edit[0] == 2:
            _, i,j = edit 
            self.__insertionInfos[i+1][0] += 1
            self.__insertionInfos[i+1][1] = min(self.__insertionInfos[i+1][1], j)
    
    def connect(self, edit:Edit, fromEdit:Optional[Edit]):
        """
        Arguments:
            - edit (Edit): the vertex to which the stop will be oriented. It will be created\
            if it doesn't exist in the graph.
            - fromEdit (Optional[Edit]): the vertex from which the stop comes. It must already\
            exist in the graph. If not specified, the empty vertex at the beginning of the graph\
            will be chosen.
            0
        Add an oriented edge from the `fromEdit` vertex to the `edit` one.
        """
        if not edit in self.__editIds:
            self.__addNode(edit)
        nodeId = self.__editIds[edit]
        fromNodeId = 0
        if fromEdit is not None:
            fromNodeId = self.__editIds[fromEdit]
        self.__linkNodes(nodeId, fromNodeId)
    
    def __linkNodes(self, toNode:int, fromNode:int):
        self.__nodes[fromNode].addChildVertex(toNode)
        self.__nodes[toNode].addParentVertex(fromNode)
    
    def getNode(self, id_:int):
        return self.__nodes[id_]
    
    def include(self, edit:Edit)->bool:
        return edit in self.__editIds
    
    def getEdit(self, nodeId:int)->ByteTensor:
        return ByteTensor(self.__edits[nodeId], device=device)
    
    @property
    def initialNode(self):
        return self.__nodes[0]
    
    def __computeInsertionIndex(self, insertion:Edit)->int:
        """
        When an insertion is applied, the inserted character is inserted at
        the position f(i) + k, with f(i) the position the i-th character
        of x in the non-contatenated proposal. This method computes k (>= 1) for a given insertion.
        """
        op, i, j = insertion
        assert(op==2), "The argument must be an insertion edit"
        return j - self.__insertionInfos[i+1][1] + 1
    
    def computeEditsCombinations(self) -> Tensor:
        """
        Computes the combinations of edits that we can do with the available edit paths\
        in this edits graph. Returns all of them in a list.
        The algorithm which is used to carry out this function is a breadth-first search.
        """
        numberOfNodes = len(self.__nodes)
        f_i:list[int] = [-1]
        nonConcatenatedProposalLength = self.__insertionInfos[0][0]
        oneHotX, oneHotY = wordToOneHots(self.__x), wordToOneHots(self.__y)
        for i in range(1, len(self.__x)+1):
            f_i.append(nonConcatenatedProposalLength)
            nonConcatenatedProposalLength += 1+self.__insertionInfos[i][0]
        combinationsByAlreadySeenNodes = [torch.zeros((0, nonConcatenatedProposalLength), dtype=uint8, device=device) for _ in range(numberOfNodes)]
        emptyCombination = torch.zeros((1, nonConcatenatedProposalLength), dtype=uint8, device=device)
        for i in range(1, len(self.__x)+1):
            emptyCombination[0, f_i[i]] = oneHotX[i-1]
        combinationsByAlreadySeenNodes[0] = torch.cat([combinationsByAlreadySeenNodes[0], 
                                emptyCombination], dim=0)
        ancestorsOfNodes = [set[int]() for _ in self.__nodes]
        nodeStack = deque([0])
        nodeWithItsLongestCombination = [True for _ in range(numberOfNodes)]
        while len(nodeStack) > 0:
            currentNodeId = nodeStack.popleft()
            currentNode = self.__nodes[currentNodeId]
            # Add childs to queue, if not already done 
            for childId in currentNode.childVertecies:
                if not childId in nodeStack:
                    nodeStack.append(childId)
            
            
            # Compute new combinations from the others computed with the node's ancestors
            
            # figure out the ancestors of the node
            for parentId in currentNode.parentVertecies:
                ancestorsOfNodes[currentNodeId] = ancestorsOfNodes[currentNodeId]\
                    .union(ancestorsOfNodes[parentId])
            for ancestorId in ancestorsOfNodes[currentNodeId]:
                combinationsByAlreadySeenNodes[currentNodeId] = torch.cat(
                    (combinationsByAlreadySeenNodes[currentNodeId],
                     combinationsByAlreadySeenNodes[ancestorId]), dim=0
                )
            # the parent(s) must be treated at the end so the final combination is the one with the
            # greatest number of applied edits
            parentsIds = list(currentNode.parentVertecies)
            if len(parentsIds)==2:
                # here a combination is removed because it generates the same proposal that another one
                if nodeWithItsLongestCombination[0] and nodeWithItsLongestCombination[1]:
                    combinationsByAlreadySeenNodes[parentsIds[0]] = combinationsByAlreadySeenNodes[parentsIds[0]][:-1, :]
                    nodeWithItsLongestCombination[parentsIds[0]] = False
                elif nodeWithItsLongestCombination[parentsIds[0]]:
                    a,b = parentsIds
                    parentsIds = [b, a]
            for parentId in parentsIds:
                combinationsByAlreadySeenNodes[currentNodeId] = torch.cat(
                    (combinationsByAlreadySeenNodes[currentNodeId],
                     combinationsByAlreadySeenNodes[parentId]), dim=0
                )
                ancestorsOfNodes[currentNodeId].add(parentId)

            
            # prepare the edit and apply it on all the duplicated combinations
            edit = self.__edits[currentNodeId]
            newChar = 0 if edit[0]==1 else oneHotY[edit[2]]
            indexWhereApplyingEdit = f_i[edit[1]+1]
            if edit[0]==2:
                indexWhereApplyingEdit += self.__computeInsertionIndex(edit)
            combinationsByAlreadySeenNodes[currentNodeId][:, indexWhereApplyingEdit] = newChar*torch.ones(combinationsByAlreadySeenNodes[currentNodeId].shape[0], dtype=uint8, device=device)
        proposals = torch.zeros((0, nonConcatenatedProposalLength), dtype=uint8, device=device)
        for n in range(numberOfNodes):
            proposals = torch.cat((proposals, combinationsByAlreadySeenNodes.pop()), dim=0)
        return proposals
    
    def displayGraph(self, filename:str):
        """
        Debugging method. Build the adjacent matrix and use graphviz to display the graph.
        Make a in-depth walk in this graph to process each node and build the matrix. 
        """
        G = graphviz.Digraph(comment=f'/{self.__x}/ to /{self.__y}/', node_attr={'style':'filled','fillcolor':'lightgoldenrod1'})
        G.attr(bgcolor="transparent")
        for node in self.__nodes:
            if node.id_==0:
                G.node('0', self.__x, _attributes={'fillcolor':'darkorange'})
            else:
                edit = self.__edits[node.id_]
                label:str
                if edit[0]==0:
                    label = f"/{self.__x[edit[1]]}/→/{self.__y[edit[2]]}/, ({edit[1]}, {edit[2]})"
                elif edit[0]==1:
                    label = f"-/{self.__x[edit[1]]}/, ({edit[1]}, {edit[2]})"
                else:
                    label = f"+/{self.__y[edit[2]]}/, ({edit[1]}, {edit[2]}), {self.__computeInsertionIndex(edit)}"
                G.node(str(node.id_), label)
        G.node(str(len(self.__nodes)), label=self.__y, _attributes={'fillcolor':'darkorange'})
        for node in self.__nodes:
            for childId in node.childVertecies:
                G.edge(str(node.id_), str(childId))
            if len(node.childVertecies)==0:
                G.edge(str(node.id_), str(len(self.__nodes)))
        lastNode = self.__nodes[0]
        G.render(filename=f'{filename}.gv', directory='./Tests/editsGraphs/',
                 format='svg')