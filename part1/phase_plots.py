import pyspiel
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.egt import utils
from open_spiel.python.egt import visualization
import matplotlib.pyplot as plt
import numpy as np
import ternary
from ternary.helpers import project_point
import matrix_games as mg

#Constructs a phase plot for the prisoners dilemma
def matrix_pd_phaseplot(size=None, fig=None):
    fig = plt.figure(figsize=(10, 10)) if fig is None else fig
    size = 111 if size is None else size
    assert isinstance(fig, plt.Figure)

    game = pyspiel.load_game("matrix_pd")
    payoff_tensor = game_payoffs_array(game)
    dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
    sub = fig.add_subplot(size, projection="2x2")
    sub.quiver(dyn,linewidth=0.1)

    sub.set_title("Phaseplot Prisoners dilemma")
    sub.set_xlabel("Player 1")
    sub.set_ylabel("Player 2")
    return sub

def matrix_mp_phaseplot(size=None,fig = None):
    fig = plt.figure(figsize=(10, 10)) if fig is None else fig
    size = 111 if size is None else size
    assert isinstance(fig, plt.Figure)

    game = pyspiel.load_game("matrix_mp")
    payoff_tensor = game_payoffs_array(game)
    dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
    sub = fig.add_subplot(size, projection="2x2")
    sub.quiver(dyn)

    sub.set_title("Phaseplot Matching pennies")
    sub.set_xlabel("Player 1")
    sub.set_ylabel("Player 2")
    return sub

def matrix_bots_phaseplot(size=None, fig=None):
    fig = plt.figure(figsize=(10, 10)) if fig is None else fig
    size = 111 if size is None else size
    assert isinstance(fig, plt.Figure)

    payoff_tensor = np.array([[[3, 0],[0, 2]],[[2, 0],[0,3]]])
    dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
    sub = fig.add_subplot(size, projection="2x2")
    sub.quiver(dyn)

    sub.set_title("Phaseplot Battle of the sexes")
    sub.set_xlabel("Man")
    sub.set_ylabel("Woman")
    return sub

def matrix_rps_phaseplot(size=None, fig=None):
    fig = plt.figure(figsize=(10, 10)) if fig is None else fig
    size = 111 if size is None else size
    assert isinstance(fig, plt.Figure)

    game = pyspiel.load_game("matrix_rps")
    payoff_tensor = game_payoffs_array(game)
    dyn = dynamics.SinglePopulationDynamics(payoff_tensor, dynamics.replicator)
    sub = fig.add_subplot(size, projection="3x3")
    sub.quiver(dyn)

    sub.set_title("Phaseplot Rock Paper Scissors")
    return sub

def matrix_rps_biased_phaseplot(size=None, fig=None):
    fig = plt.figure(figsize=(10, 10)) if fig is None else fig
    size = 111 if size is None else size
    assert isinstance(fig, plt.Figure)

    payoff_tensor = np.array([[[0,-1,2],[1,0,-1],[-2,1,0]],[[0,1,-2],[-1,0,1],[2,-1,0]]])
    dyn = dynamics.SinglePopulationDynamics(payoff_tensor, dynamics.replicator)
    sub = fig.add_subplot(size, projection="3x3")
    sub.quiver(dyn)

    sub.set_title("Phaseplot Rock Paper Scissors")
    return sub,pyspiel.create_matrix_game(payoff_tensor[0],payoff_tensor[1])


if __name__ == "__main__":
    fig = plt.figure(figsize=(10,10))
    matrix_pd_phaseplot(221,fig)
    matrix_mp_phaseplot(222,fig)
    matrix_bots_phaseplot(223,fig)
    matrix_rps_phaseplot(224,fig)
    # matrix_rps_biased_phaseplot()
    plt.show()
