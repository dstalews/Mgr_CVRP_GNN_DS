from collections import namedtuple
from helpers.beam_monte_carlo_tree_search import BMCTS, Node
from helpers.monte_carlo_tree_search import MCTS
from helpers.beamSearch import get_best_route_bs, score_length
from helpers.cvrproute import CVRPRoute
import copy
from random import choice


 
_CVRPB = namedtuple("CVRP_route", "obj y_dist y_c k terminal")

# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class CVRPBoard(_CVRPB, Node):
    def find_children(board):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make only possible moves
        return {
            board.make_move(i) for i in range(0, board.obj.num_nodes) if board.obj.check_next_node(i, board.y_c[i])
        }

    def find_random_child(board): 
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        avalible_moves = [i for i in range(0, board.obj.num_nodes) if board.obj.check_next_node(i, board.y_c[i])]
        return board.make_move(choice(avalible_moves))

    def simulate_reward(board):
        d, l = get_best_route_bs(board.y_dist, board.y_c, board.k, board.obj)
        return -l

    def last(board):
        return board.route[-1]

    def prior_p(board):
        return board.obj.prior_p

    def reward(board): # change reward
        return -score_length(board.y_dist, board.obj.croute)

    def is_terminal(board):
        return board.terminal

    def make_move(board, index): 
        obj2 = copy.deepcopy(board.obj)
        obj2.add_node(index, board.y_c[index])
        is_terminal = obj2.is_terminal_route()
        return CVRPBoard(obj2, board.y_dist, board.y_c, board.k, is_terminal)
    
    def __hash__(board):
       return hash(board.obj)
    
    def depth(board):
        return len(board.obj.croute) - 1
    
    def croute(board):
        return board.obj.croute

def play_game(p, d, c, width, type = 'M', beam = 8, simlimit = 40):
    if type == 'B':
        tree = BMCTS(beam, simlimit)
    else:
        tree = MCTS()
    if len(c)>30:
        roll = 1200
    else:
        roll = 800
    board = new_cvrp_board(p, d, c, width)
    for i in range(roll):
        stop = tree.do_rollout(board)
        if stop == -1:
            print(i)
            break
    while True:
        board = tree.choose(board)      
        #print(board.route)
        if board.terminal:
            return board.obj.croute, score_length(board.y_dist, board.obj.croute), tree.sel_time, tree.exp_time, tree.sim_time, tree.backp_time


def new_cvrp_board(p, d, c, width): 
    n = len(c)
    return CVRPBoard(obj=CVRPRoute(n, copy.deepcopy(p)), y_dist=tuple(tuple(sub) for sub in d), y_c = c, k = width, terminal = False)
