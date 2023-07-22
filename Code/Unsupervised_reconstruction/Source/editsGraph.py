from typing import Optional
from collections import deque

from Types.models import *
from data.vocab import wordToOneHots, reduceOneHotTensor, oneHotsToWord

import torch
from torch import Tensor, uint8, ByteTensor

import graphviz

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Node:
    def __init__(self, id_: int) -> None:
        self.__id = id_
        # ids of vertecies from which the node can come
        self.__parentVertecies: set[int] = set()
        # ids of vertecies directly joinable from the node
        self.__childVertecies: set[int] = set()

    @property
    def id_(self):
        return self.__id

    @property
    def childVertecies(self):
        return self.__childVertecies

    @property
    def parentVertecies(self):
        return self.__parentVertecies

    def addChildVertex(self, vertexId: int):
        self.__childVertecies.add(vertexId)

    def removeChildVertex(self, vertexId: int):
        self.__childVertecies.remove(vertexId)

    def addParentVertex(self, vertexId: int):
        self.__parentVertecies.add(vertexId)

    def removeParentVertex(self, vertexId: int):
        self.__parentVertecies.remove(vertexId)


class EditsGraph:
    """
    This class manages the features of an oriented graph to represent
    the edit paths.
    """

    def __init__(self, x: Tensor, y: Tensor, editDistance: int) -> None:
        self.__editIds: dict[Edit, int] = {}
        self.__edits: list[Edit] = [(0, 0, 0)]
        self.__insertionInfos: list[list[int]] = [
            [0, len(y)] for i in range(len(x)+1)]
        """
        The matrix above contains for each i position in x the following information:\\
        [the max number of inserted characters, the minimal j index in y of the inserted characters]
        """
        self.__deletionInfos: list[bool] = [False for i in range(len(x))]
        """
        This list save if there is an edit path where x[i] is deleted.
        """
        # The dict below saves the graph's vertecies and their connexions.
        # Empty beginning node with 0 index
        self.__nodes: list[Node] = [Node(0)]
        self.__nodesDepth: list[int] = [0]  # list of depth of each node
        self.__editDistance = editDistance
        # For debugging
        self.__x = x
        self.__y = y

    def __addNode(self, edit: Edit, fromNodeId: int):
        newNodeId = len(self.__nodes)
        self.__edits.append(edit)
        self.__nodes.append(Node(newNodeId))
        self.__nodesDepth.append(self.__nodesDepth[fromNodeId]+1)
        self.__editIds[edit] = newNodeId
        if edit[0] == 2:
            _, i, j = edit
            self.__insertionInfos[i+1][0] += 1
            self.__insertionInfos[i +
                                  1][1] = min(self.__insertionInfos[i+1][1], j)
        elif edit[0] == 1:
            i = edit[1]
            self.__deletionInfos[i] = True

    def connect(self, edit: Edit, fromEdit: Optional[Edit]):
        """
        Add an oriented edge from the `fromEdit` vertex to the `edit` one.

        Arguments:
            - edit (Edit): the vertex to which the stop will be oriented. It will be created\
            if it doesn't exist in the graph.
            - fromEdit (Optional[Edit]): the vertex from which the stop comes. It must already\
            exist in the graph. If not specified, the empty vertex at the beginning of the graph\
            will be chosen.
            0
        """
        fromNodeId = 0

        if fromEdit is not None:
            fromNodeId = self.__editIds[fromEdit]

        if not edit in self.__editIds:
            self.__addNode(edit, fromNodeId)

        nodeId = self.__editIds[edit]
        self.__nodesDepth[nodeId]
        self.__linkNodes(nodeId, fromNodeId)

    def __linkNodes(self, toNode: int, fromNode: int):
        self.__nodes[fromNode].addChildVertex(toNode)
        self.__nodes[toNode].addParentVertex(fromNode)

    def getNode(self, id_: int):
        return self.__nodes[id_]

    def getLastNode(self) -> Node:
        lastNodes: list[Node] = []

        for node in self.__nodes:
            if len(node.childVertecies) == 0:
                lastNodes.append(node)

        if len(lastNodes) == 1:
            return lastNodes[0]

        elif len(lastNodes) == 2:
            nodeToBeReturned = Node(-1)

            for node in lastNodes:
                nodeToBeReturned.addParentVertex(node.id_)
            return nodeToBeReturned

        else:
            raise Exception('It must be one or two last node(s).')

    def include(self, edit: Edit) -> bool:
        return edit in self.__editIds

    def getEdit(self, nodeId: int) -> ByteTensor:
        return ByteTensor(self.__edits[nodeId], device=device)

    @property
    def initialNode(self):
        return self.__nodes[0]

    def __computeInsertionIndex(self, insertion: Edit) -> int:
        """
        When an insertion is applied, the inserted character is inserted at
        the position f(i) + k, with f(i) the position the i-th character
        of x in the non-contatenated proposal. This method computes k (>= 1) for a given insertion.
        """
        op, i, j = insertion
        assert (op == 2), "The argument must be an insertion edit"
        return j - self.__insertionInfos[i+1][1] + 1

    def __rollTensor(self, t: Tensor, j: int):
        t[:, j:] = torch.where((t[:, j] == 0).unsqueeze(1),
                               (t[:, j:]).roll(-1, 1),
                               t[:, j:])

    def __moveZeros(self, t: Tensor) -> Tensor:
        """
        Rewrite each tensor's row for the zeros to all be on the right.

        Arguments:
            t (Tensor): a tensor of shape (batch_size, N)
        """
        N = t.shape[1]

        for j in range(N-2, -1, -1):
            self.__rollTensor(t, j)

        # DEBUG
        # for j in range(N):
        #     if torch.all(t[:,j]==0).item():
        #         for k in range(j+1, N):
        #             if not torch.all(t[:,k]==0).item():
        #                 raise Exception("The algorithm is wrong.")
        #         break
        return reduceOneHotTensor(t)

    def __addCombinations(self, combinationsList: list[Tensor], fromNode_id: int, toNode_id: int):
        combinationsList[toNode_id] = torch.cat(
            (combinationsList[toNode_id],
             combinationsList[fromNode_id]), dim=0
        )

    def __nodesWithAllCombinations(self, ) -> list[bool]:
        """
        All the edits graph is crossed to figure out for each node whether the combinations will be computed from its parent.
        Sometimes, it does not have to happen, in the order to prevent the duplication of generated proposals.
        """
        combinateFromTheParent = [True for _ in range(len(self.__nodes))]
        # TODO
        return combinateFromTheParent

    def computeEditsCombinations(self) -> Tensor:
        """
        Computes the combinations of edits that we can do with the available edit paths\
        in this edits graph. Returns all of them in a list.
        The algorithm which is used to carry out this function is a breadth-first search.
        """
        numberOfNodes = len(self.__nodes)
        f_i: list[int] = [-1]
        nonConcatenatedProposalLength = self.__insertionInfos[0][0]
        oneHotX, oneHotY = self.__x, self.__y

        for i in range(1, len(self.__x)+1):
            f_i.append(nonConcatenatedProposalLength)
            nonConcatenatedProposalLength += 1+self.__insertionInfos[i][0]

        combinationsByAlreadySeenNodes = [torch.zeros(
            (0, nonConcatenatedProposalLength), dtype=uint8, device=device) for _ in range(numberOfNodes)]

        emptyCombination = torch.zeros(
            (1, nonConcatenatedProposalLength), dtype=uint8, device=device)

        for i in range(1, len(self.__x)+1):
            emptyCombination[0, f_i[i]] = oneHotX[i-1]

        combinationsByAlreadySeenNodes[0] = torch.cat([combinationsByAlreadySeenNodes[0],
                                                       emptyCombination], dim=0)

        ancestorsOfNodes = [set[int]() for _ in self.__nodes]
        nodeStack = deque([0])

        # Breadth-first search
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
                ancestorsOfNodes[currentNodeId].add(parentId)

            interestingAncestors = ancestorsOfNodes[currentNodeId].copy()

            for ancestorId in interestingAncestors:
                self.__addCombinations(
                    combinationsByAlreadySeenNodes, ancestorId, currentNodeId)

            # prepare the edit and apply it on all the selected duplicated combinations
            edit = self.__edits[currentNodeId]
            newChar = 0 if edit[0] == 1 else oneHotY[edit[2]].item()
            indexWhereApplyingEdit = f_i[edit[1]+1]

            if edit[0] == 2:
                indexWhereApplyingEdit += self.__computeInsertionIndex(edit)

            combinationsByAlreadySeenNodes[currentNodeId][:, indexWhereApplyingEdit] = newChar*torch.ones(
                combinationsByAlreadySeenNodes[currentNodeId].shape[0], dtype=uint8, device=device)

        # Gather all the computed combinations and translate all the sparsed zeros to the right of the proposals matrix
        proposals = torch.zeros(
            (0, nonConcatenatedProposalLength), dtype=uint8, device=device)

        for _ in range(numberOfNodes):
            proposals = torch.cat(
                (proposals, combinationsByAlreadySeenNodes.pop()), dim=0)

        proposals = self.__moveZeros(proposals)
        return proposals

    def displayGraph(self, filename: str):
        """
        Debugging method. Build the adjacent matrix and use graphviz to display the graph.
        Make a in-depth walk in this graph to process each node and build the matrix. 
        """
        x, y = oneHotsToWord(self.__x), oneHotsToWord(self.__y)
        G = graphviz.Digraph(comment=f'/{x}/ to /{y}/', node_attr={
                             'style': 'filled', 'fillcolor': 'lightgoldenrod1'})
        G.attr(bgcolor="transparent")
        c = self.__nodesWithAllCombinations()

        for nodeId, node in enumerate(self.__nodes):
            if nodeId == 0:
                G.node('0', x, _attributes={'fillcolor': 'darkorange'})
            else:
                edit = self.__edits[node.id_]
                label: str
                if edit[0] == 0:
                    label = f"/{x[edit[1]]}/→/{y[edit[2]]}/, ({edit[1]}, {edit[2]})"
                elif edit[0] == 1:
                    label = f"-/{x[edit[1]]}/, ({edit[1]}, {edit[2]})"
                else:
                    label = f"+/{y[edit[2]]}/, ({edit[1]}, {edit[2]}), {self.__computeInsertionIndex(edit)}"
                if c[nodeId]:
                    G.node(str(node.id_), label)
                else:
                    G.node(str(node.id_), label, _attributes={
                           'fillcolor': 'lightcoral'})

        G.node(str(len(self.__nodes)), label=y,
               _attributes={'fillcolor': 'darkorange'})

        for node in self.__nodes:
            for childId in node.childVertecies:
                G.edge(str(node.id_), str(childId))

            if len(node.childVertecies) == 0:
                G.edge(str(node.id_), str(len(self.__nodes)))

        G.render(filename=f'{filename}.gv', directory='./Tests/editsGraphs/',
                 format='svg')
