import open_spiel.games
import pyspiel
from tabulate import tabulate
import numpy as np

def pretty_print_strategies(game,strategies):
    if not isinstance(game,pyspiel.MatrixGame):
        raise Exception("Game " + game + "is not a matrix game.")

    if not len(strategies.shape) == 2:
        raise Exception("Strategies not correct shape, dimension must be exactly 2, got "+ str(len(strategies.shape)))

    action_names = [game.col_action_name(i) for i in range(game.num_cols())]
    strats = tabulate(strategies, headers=action_names)
    print(strats)

game = pyspiel.load_game("matrix_pd")

