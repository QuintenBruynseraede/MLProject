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

def matrix_pd_phaseplot(size,learning_rate):
    game = pyspiel.load_game("matrix_pd")
    payoff_matrix = game_payoffs_array(game)
    dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)


    xy = np.mgrid[0:1.1:1/size, 0:1.1:1/size].reshape(2, -1).T
    for p in xy:
        res = p + learning_rate * dyn(p)
        plt.arrow(p[0],p[1],(res-p)[0],(res-p)[1], head_width=0.01)

    plt.xticks([0,1],["Cooperate","Defect"])
    plt.yticks([0,1],["Cooperate","Defect"])
    plt.xlabel("Player 1")
    plt.ylabel("Player 2")
    plt.show()




def train_algorithm(algorithm, game):
    env = rl_environment.Environment(game)

def plot_phase_ternary(actions, arrows=True,arrows_every=2, title="Phase diagram"):
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
    matrix_pd_phaseplot(20,0.01)

    #plt.plot(actions)
    plt.show()

