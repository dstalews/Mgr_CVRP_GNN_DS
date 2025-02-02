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

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
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
            if self.N[hash(n)] == 0:
                return float("-inf")  # avoid unseen moves
            return self.P(n)

        return max(self.children[hash(node)], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        t = time.time()
        path = self._select(node)
        self.sel_time += time.time()-t
        t = time.time()
        leaf = path[-1] #expand with best propability
        self._expand(leaf)
        self.exp_time += time.time()-t
        t = time.time()
        reward = self._simulate(leaf)
        self.sim_time += time.time()-t
        t = time.time()
        self._backpropagate(path, reward)
        self.backp_time += time.time()-t

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if hash(node) not in self.children or not self.children[hash(node)]:
                # node is either unexplored or terminal
                return path
            for n in self.children[hash(node)]:
                if hash(n) not in self.children.keys():
                    path.append(n)
                    return path
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