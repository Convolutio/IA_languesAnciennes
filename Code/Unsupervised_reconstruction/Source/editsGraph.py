from Types.models import *
from bidict import bidict
from typing import Optional
import graphviz
from collections import deque

def addEditToCombi(combi:EditsCombination, edit:EditNDArray):
    numberOfCombination = combi.item(0,0)
    combi[numberOfCombination] = edit
    combi[0,0] = numberOfCombination + 1
    return combi
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

class EditsGraph:
    """
    This class manages the features of an oriented graph to represent
    the edit paths.
    """
    def __init__(self, x:str, y:str, editDistance:int) -> None:
        self.__editIds:dict[Edit, int] = {}
        self.__editsNDArrays = np.array([[0,0,0,0]], dtype=np.int8)
        # The dict below saves the graph's vertecies and their connexions.
        self.__nodes:list[Node] = [Node(0)] # Empty beginning node with 0 index
        self.__editDistance = editDistance
        #For debugging
        self.__x = x
        self.__y = y

    def __addNode(self, edit:Edit):
        newNodeId = len(self.__nodes)
        self.__editsNDArrays = np.append(arr=self.__editsNDArrays, values=np.array([edit], dtype=np.int8), axis=0)
        self.__nodes.append(Node(newNodeId))
        self.__editIds[edit] = newNodeId
    
    def connect(self, edit:Edit, fromEdit:Optional[Edit]):
        """
        Arguments:
            - edit (Edit): the vertex to which the stop will be oriented. It will be created\
            if it doesn't exist in the graph.
            - fromEdit (Optional[Edit]): the vertex from which the stop comes. It must already\
            exist in the graph. If not specified, the empty vertex at the beginning of the graph\
            will be chosen.
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
    
    def getEdit(self, nodeId:int)->npt.NDArray[np.int8]:
        return self.__editsNDArrays[nodeId]
    
    @property
    def initialNode(self):
        return self.__nodes[0]
    
    def computeEditsCombinations(self) -> npt.NDArray[np.int8]:
        """
        Computes the combinations of edits that we can do with the available edit paths\
        in this edits graph. Returns all of them in a list.
        The algorithm which is used to carry out this function is a breadth-first search.
        """
        numberOfNodes = len(self.__nodes)
        #The first combination bring the length information : [length, 0, 0, 0]
        combinationsByAlreadySeenNodes = [np.zeros((0, 1+self.__editDistance, 4), dtype=np.int8) for _ in range(numberOfNodes)]
        combinationsByAlreadySeenNodes[0] = np.append(combinationsByAlreadySeenNodes[0],
                                                   np.zeros((1,1+self.__editDistance,4),dtype=np.int8), 0)
        ancestorsOfNodes = [set[int]() for _ in self.__nodes]
        nodeStack = deque([0])
        while len(nodeStack) > 0:
            currentNodeId = nodeStack.popleft()
            currentNode = self.__nodes[currentNodeId]
            # figure out the ancestors of the node
            for parentId in currentNode.parentVertecies:
                ancestorsOfNodes[currentNodeId] = ancestorsOfNodes[currentNodeId]\
                    .union(ancestorsOfNodes[parentId])
                ancestorsOfNodes[currentNodeId].add(parentId)
            # Add childs to queue, if not already done 
            for childId in currentNode.childVertecies:
                if not childId in nodeStack:
                    nodeStack.append(childId)
            # Compute new combinations from the other computed with the node's ancestors
            for ancestorId in ancestorsOfNodes[currentNodeId]:
                combinationsByAlreadySeenNodes[currentNodeId] = np.append(combinationsByAlreadySeenNodes[currentNodeId],
                                                                          combinationsByAlreadySeenNodes[ancestorId], 0)
            if currentNodeId != 0:
                for i in range(combinationsByAlreadySeenNodes[currentNodeId].shape[0]):
                    oldcombinationsNumber = combinationsByAlreadySeenNodes[currentNodeId].item((i, 0, 0))
                    combinationsByAlreadySeenNodes[currentNodeId][i, oldcombinationsNumber+1] = self.__editsNDArrays[currentNodeId]
                    combinationsByAlreadySeenNodes[currentNodeId][i, 0, 0] = oldcombinationsNumber + 1
        
        combinArray = np.zeros((0,1+self.__editDistance, 4), dtype=np.int8)
        for n in range(numberOfNodes):
            combinArray = np.append(combinArray, combinationsByAlreadySeenNodes.pop(), 0)
        return combinArray
    
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
                edit:Edit = tuple(self.__editsNDArrays[node.id_])
                label = f"+/{self.__y[edit[2]]}/, ({edit[1]}, {edit[2]}), {edit[3]}"
                if edit[0]==0:
                    label = f"/{self.__x[edit[1]]}/→/{self.__y[edit[2]]}/, ({edit[1]}, {edit[2]})"
                elif edit[0]==1:
                    label = f"-/{self.__x[edit[1]]}/, ({edit[1]}, {edit[2]})"
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