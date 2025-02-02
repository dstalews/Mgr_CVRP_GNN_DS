"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import time

class BMCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, beamwidth, simlimit, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.beamwidth = beamwidth
        self.simlimit = simlimit
        self.numberOfRollouts = defaultdict(int)  # number of rollouts on each depth
        self.sim_time = 0
        self.sel_time = 0
        self.exp_time = 0
        self.backp_time = 0

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if hash(node) not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[hash(n)] == 0 or hash(n) not in self.children.keys():
                return float("-inf")  # avoid unseen moves
            return self.P(n)

        return max(self.children[hash(node)], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        t = time.time()
        path = self._select(node)
        self.sel_time += time.time()-t
        t = time.time()
        if path == []:
            return -1
        leaf = path[-1] #expand with best propability
        self._expand(leaf)
        self.exp_time += time.time()-t
        t = time.time()
        reward = self._simulate(leaf)
        self.sim_time += time.time()-t
        t = time.time()
        self._backpropagate(path, reward)
        if self.numberOfRollouts[leaf.depth()] >= self.simlimit:                                                             
            self.children = self.pruneTree2([node], leaf.depth(), node.depth())
        self.backp_time += time.time()-t    
        return 0

# przerobić na sklejenie nowego drzewa, zamiast modyfikacji starego
    def pruneTree(self, node_list, depthLimit, nodeDepth): 
        new_node_list = []
        new_node_depth = nodeDepth + 1
        for node in node_list:
            if hash(node) in self.children.keys():
                for n in self.children[hash(node)]:
                    if n not in new_node_list:
                        new_node_list.append(n)
        if new_node_depth < depthLimit:
            return self.pruneNodes(self.pruneTree(new_node_list, depthLimit, new_node_depth), node_list)
        else:
            ordered = sorted(new_node_list, key=lambda x: self.Q[hash(x)], reverse=True)
            return self.pruneNodes(ordered[self.beamwidth:], node_list)
        
    def pruneNodes(self, nodes_to_prune, ancestor_nodes):
        new_nodes_to_prune = []
        for nn in nodes_to_prune:
            if hash(nn) in self.children.keys():
                del self.children[hash(nn)]
        for n in ancestor_nodes:
            if hash(n) in self.children.keys():
                for nn in nodes_to_prune:
                    self.children[hash(n)].discard(nn)
                if not len(self.children[hash(n)]):
                    new_nodes_to_prune.append(n)
        return new_nodes_to_prune       
    
    # przerobić na sklejenie nowego drzewa, zamiast modyfikacji starego
    def pruneTree2(self, node_list, depthLimit, nodeDepth): 
        new_node_list = []
        new_node_depth = nodeDepth + 1
        for node in node_list:
            if hash(node) in self.children.keys():
                for n in self.children[hash(node)]:
                    if n not in new_node_list and self.Q[hash(n)] != 0:
                        new_node_list.append(n)
        if new_node_depth < depthLimit:
            return self.generateTree(node_list, self.pruneTree2(new_node_list, depthLimit, new_node_depth))
        else:
            tree = defaultdict(set)
            ordered = sorted(new_node_list, key=lambda x: self.Q[hash(x)], reverse=True)[:self.beamwidth]
            for n in ordered:
                if hash(n) in self.children.keys():
                    tree[hash(n)] = self.children[hash(n)]
            return self.generateTree(node_list, tree)
            
    def generateTree(self, ancestor_nodes, tree):
        for n in ancestor_nodes:
            for nn in self.children[hash(n)]:
                if hash(nn) in tree.keys():
                    tree[hash(n)].add(nn)
        return tree

    def _select(self, node):

        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if hash(node) not in self.children or not self.children[hash(node)]:
                # node is either unexplored or terminal
                self.numberOfRollouts[node.depth()]+=1
                return path
            for n in self.children[hash(node)]:
                if hash(n) not in self.children.keys():
                    self.numberOfRollouts[n.depth()]+=1
                    path.append(n)
                    return path
            if node.is_terminal():
                return [] 
            node = self._uct_select(node)  # descend a layer deeper


    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if hash(node) in self.children:
            return  # already expanded
        self.children[hash(node)] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        return node.simulate_reward()
    
    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            if self.N[hash(node)] == 0 or self.Q[hash(node)] < reward:
                self.Q[hash(node)] = reward
            self.N[hash(node)] += 1

    def sum_Q(self, node):
        sum = 0
        for x in self.children[hash(node)]:
            sum += self.Q[hash(x)]
        if sum == 0:
            return 1
        return sum

    def sum_N(self, node):
        sum = 0
        for x in self.children[hash(node)]:
            sum += self.N[hash(x)]
        return sum

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        #assert all(n in self.children for n in self.children[hash(node)])

        # Upper confidence bound from "A Graph Neural Network Assisted Monte
        # Carlo Tree Search Approach to Traveling Salesman Problem"
        # def uct(n):
        #     "Upper confidence bound for trees"
        #     return self.Q[hash(n)] + self.exploration_weight * n.prior_p() * math.sqrt(
        #         self.sum_N(n)) / (self.N[hash(n)] + 1)
        

        log_N_vertex = math.log(self.N[hash(node)])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[hash(n)] / self.N[hash(n)] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[hash(n)]
            )

        return max(self.children[hash(node)], key=uct)

    def P(self, n):
        return 1 + self.Q[hash(n)]/self.sum_Q(n)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None
    
    @abstractmethod
    def simulate_reward(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
    
    @abstractmethod
    def last(self):
        return 0
    
    @abstractmethod
    def prior_p(self):
        return 0
    
    @abstractmethod
    def depth(self):
        return 0
    
    @abstractmethod
    def croute(self):
        return 0