from Types.models import Edit
from bidict import bidict
from typing import Optional
import numpy as np

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
    def __init__(self) -> None:
        self.__nextNodeId = 1
        self.__bimaps: bidict[int, Edit] = bidict() # maps bidirectionnaly the edit and its vertex id
        # The dict below saves the graph's vertecies and their connexions.
        self.__nodes:dict[int, Node] = {0:Node(0)} # Empty beginning node with 0 index

    def __addNode(self, edit:Edit):
        self.__bimaps[self.__nextNodeId] = edit
        self.__nodes[self.__nextNodeId] = Node(self.__nextNodeId)
        self.__nextNodeId += 1
    
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
        if not edit in self.__bimaps.inverse:
            self.__addNode(edit)
        nodeId = self.__bimaps.inverse[edit]
        fromNodeId = 0
        if fromEdit is not None:
            fromNodeId = self.__bimaps.inverse[fromEdit]
        self.__linkNodes(nodeId, fromNodeId)
    
    def __linkNodes(self, toNode:int, fromNode:int):
        self.__nodes[fromNode].addChildVertex(toNode)
        self.__nodes[toNode].addParentVertex(fromNode)
    
    def __unlinkNodes(self, toNode: int, fromNode:int):
        self.__nodes[fromNode].removeChildVertex(toNode)
        self.__nodes[toNode].removeParentVertex(fromNode)

    def disconnect(self, edit:Edit, fromEdit:Edit):
        editId, fromEditId = self.__bimaps.inverse[edit], self.__bimaps.inverse[fromEdit]
        self.__unlinkNodes(editId, fromEditId)

    def __removeNode(self, id_:int):
        node = self.__nodes[id_]
        for childId in node.childVertecies:
            self.__unlinkNodes(childId, id_)
        for parentId in node.parentVertecies:
            self.__unlinkNodes(id_, parentId)
        self.__nodes.pop(id_)
        self.__bimaps.pop(id_)

    def dfs(self):
        #TODO
        pass
    
    def displayGraph(self, currentNodeIdx=None, matrix = None):
        """
        Debugging method. Build the adjacent matrix and use graphviz to display the graph.
        Make a in-depth walk in this graph to process each node and build the matrix. 
        """
        #TODO
        currentNode = self.__nodes[0]
        if currentNodeIdx is not None:
            currentNode = self.__nodes[currentNodeIdx]
        if matrix is None:
            matrix = np.zeros((len(self.__nodes), len(self.__nodes)), dtype=int)
        pass