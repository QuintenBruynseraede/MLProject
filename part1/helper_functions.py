import open_spiel.games
import pyspiel
from tabulate import tabulate
import numpy as np
import math

def pretty_print_strategies(game,strategies):
    if not isinstance(game,pyspiel.MatrixGame):
        raise Exception("Game " + game + "is not a matrix game.")

    if not len(strategies.shape) == 2:
        raise Exception("Strategies not correct shape, dimension must be exactly 2, got "+ str(len(strategies.shape)))

    action_names = [game.col_action_name(i) for i in range(game.num_cols())]
    strats = tabulate(strategies, headers=action_names)
    print(strats)

game = pyspiel.load_game("matrix_pd")

def dist(p1,p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


