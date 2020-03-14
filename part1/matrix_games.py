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
def get_replicator_dynamics(game_name, iterations,learning_rate,verbose=False):
    game = pyspiel.load_game(game_name)
    if not isinstance(game,pyspiel.MatrixGame):
        print("Game " + game_name + "is not a matrix game, construction of payoff matrix will take a long time...")
    payoff_matrix = game_payoffs_array(game)
    dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)

    legal_actions = game.num_distinct_actions()
    #x = [np.random.rand() for _ in range(legal_actions)]
    x = np.array([0.5,0.2,0.3])
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

def plot_ternary(actions, arrows=True,arrows_every=4, title="Phase diagram"):
    assert actions.shape[1] == 3

    _, tax = ternary.figure()
    tax.set_title(title, fontsize=15)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=0.05, color="gray")
    tax.scatter(actions, color='blue', label="Actions", s = 0.1)
    ax = tax.get_axes()

    for i in range(len(actions)-1):
        if i % arrows_every == 0:
            p1 = project_point(actions[i])
            p2 = project_point(actions[i+1])
            ax.arrow(p1[0], p1[1], (p2-p1)[0],(p2-p1)[1],head_width=0.005, head_length=0.005, fc='k', ec='k')    
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    tax.ticks(axis='lbr', linewidth=1, multiple=5, offset=0.03)
    tax.show()


if __name__ == "__main__":
    actions = get_replicator_dynamics("matrix_rps", 10000,0.1,verbose=False)
    plot_ternary(actions, arrows_every=2, title="Rock Paper Scissors")
    #plt.plot(actions)
    plt.show()