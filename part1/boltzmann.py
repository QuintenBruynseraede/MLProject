import pyspiel
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.egt import utils
from open_spiel.python.egt import visualization
import matplotlib.pyplot as plt
import numpy as np
import math


#Constructs a phase plot for the prisoners dilemma
def matrix_pd_phaseplot(size=None, fig=None):
    fig = plt.figure(figsize=(10, 10)) if fig is None else fig
    size = 111 if size is None else size
    assert isinstance(fig, plt.Figure)

    game = pyspiel.load_game("matrix_pd")
    payoff_tensor = game_payoffs_array(game)
    dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
    print(dyn)
    sub = fig.add_subplot(size, projection="2x2")
    sub.quiver(dyn)

    sub.set_title("Phaseplot Prisoners dilemma")
    sub.set_xlabel("Player 1")
    sub.set_ylabel("Player 2")
    return sub


def pd_phaseplot_boltzmann(fig):
    #Load game an payoff matrices
    game = pyspiel.load_game("matrix_pd")
    payoff_tensor = game_payoffs_array(game)
    A = payoff_tensor[0]
    B = payoff_tensor[1]
    size = 111


    game = pyspiel.load_game("matrix_pd")
    payoff_tensor = game_payoffs_array(game)
    dyn = dynamics.MultiPopulationDynamics(payoff_tensor, lenient_boltzmannq)
    sub = fig.add_subplot(size, projection="2x2")
    sub.quiver(dyn)


    sub.set_title("Phaseplot Prisoners dilemma")
    sub.set_xlabel("Player 1")
    sub.set_ylabel("Player 2")
    return sub

def lenient_boltzmannq(state, fitness):
    temperature = 1
    kappa = 6
    A = np.array([[ 5., 0.],[10., 1.]])
    x = list()
    y = list()
    x.append(fitness[0])
    x.append(1-fitness[0])
    y.append(fitness[1])
    y.append(1-fitness[1])
    #a i,j = A[i][j]   
    fitness_exploitation = list()
    for i in range(len(fitness)):
        #j=0
        term0 = A[i][0] * y[0] * ( sum([y[k] for k in range(2) if A[i][k] <= A[i][0]]) ** kappa - sum([y[k] for k in range(2) if A[i][k] < A[i][0]]) ** kappa) / sum([y[k] for k in range(2) if A[i][k] == A[i][0]])
        #j=1
        term1 = A[i][1] * y[1] * ( sum([y[k] for k in range(2) if A[i][k] <= A[i][1]]) ** kappa - sum([y[k] for k in range(2) if A[i][k] < A[i][1]]) ** kappa) / sum([y[k] for k in range(2) if A[i][k] == A[i][1]])
        fitness_exploitation.append(term0+term1)
    
    exploitation = (1. / temperature) * dynamics.replicator(state, fitness_exploitation)
    exploration = (np.log(state) - state.dot(np.log(state).transpose()))
    return exploitation - state * exploration

if __name__ == "__main__":
    fig = plt.figure(figsize=(10,10))
    pd_phaseplot_boltzmann(fig)
    plt.show()

