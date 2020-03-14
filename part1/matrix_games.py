from open_spiel.python import rl_environment
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
import matplotlib.pyplot as plt
import pyspiel
import numpy as np
import helper_functions as util
import ternary
from ternary.helpers import project_point

# @param game: the game to be simulated
def get_replicator_dynamics(game_name, iterations,learning_rate=0.01,verbose=False):
    game = pyspiel.load_game(game_name)
    if not isinstance(game,pyspiel.MatrixGame):
        print("Game " + game_name + "is not a matrix game, construction of payoff matrix will take a long time...")
    payoff_matrix = game_payoffs_array(game)
    dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)

    legal_actions = game.num_distinct_actions()
    x = [np.random.rand() for _ in range(legal_actions)]
    s = sum(x)
    x = [v / s for v in x]

    actions = np.zeros((iterations,legal_actions))
    for i in range(0, iterations):
        x += learning_rate * dyn(x)
        actions[i] = x

    if verbose:
        util.pretty_print_strategies(game,actions)

    return actions


def train_algorithm(algorithm, game):
    env = rl_environment.Environment(game)


