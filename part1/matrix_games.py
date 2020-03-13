from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

# @param game: the game to be simulated
def get_replicator_dynamics(game, iterations):
    env = rl_environment.Environment(game)
    payoff_matrix = game_payoffs_array(env.game)
    dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)

    legal_actions = env.game.num_distinct_actions()
    x = [np.random.rand() for _ in range(legal_actions)]
    s = sum(x)
    x = [v / s for v in x]
    print("Initial population: " + str(x))

    actions = np.zeros((iterations,legal_actions))
    alpha = 10/iterations
    for i in range(0, iterations):
        x += alpha * dyn(x)
        actions[i] = x
    return actions


def simula1

actions = get_replicator_dynamics("matrix_", 1000)
plt.plot(actions)
plt.show()