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
    #Load game and payoff matrices
    size = 111
    game = pyspiel.load_game("matrix_mp")
    A = game_payoffs_array(game)[0]
    print(A)
    payoff_tensor = game_payoffs_array(game)
    dyn = dynamics.MultiPopulationDynamics(payoff_tensor, lenient_boltzmannq)
    sub = fig.add_subplot(size, projection="2x2")
    sub.quiver(dyn)

    for _ in range(5):
        p_x = np.random.rand()
        p_y = np.random.rand()
        x = np.array([p_x, 1-p_x, p_y, 1-p_y])
        xlist = list()
        ylist = list()
        alpha = 0.01
        iterations = 1000
        for _ in range(iterations):
            x += alpha*dyn(x)
            xlist.append(x[0])
            ylist.append(x[2])

        sub.plot(xlist,ylist)

    sub.set_title("Phaseplot Prisoners dilemma")
    sub.set_xlabel("Player 1")
    sub.set_ylabel("Player 2")
    return sub

def lenient_boltzmannq(state, fitness):
    temperature = 1
    kappa = 10
    
    #PD
    #A = np.array([[ 5, 0],[10, 1]])
    #MP
    A = np.array([[ 1, -1],[-1, 1]])
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
        term0 = A[i][0] * y[0] * ( math.pow(sum([y[k] for k in range(2) if A[i][k] <= A[i][0]]),kappa) - math.pow(sum([y[k] for k in range(2) if A[i][k] < A[i][0]]),kappa)) / sum([y[k] for k in range(2) if A[i][k] == A[i][0]])
        #j=1
        term1 = A[i][1] * y[1] * ( math.pow(sum([y[k] for k in range(2) if A[i][k] <= A[i][1]]),kappa) - math.pow(sum([y[k] for k in range(2) if A[i][k] < A[i][1]]),kappa)) / sum([y[k] for k in range(2) if A[i][k] == A[i][1]])
        fitness_exploitation.append(term0+term1)
    
    exploitation = (1. / temperature) * dynamics.replicator(state, fitness_exploitation)
    exploration = (np.log(state) - state.dot(np.log(state).transpose()))
    return exploitation - state * exploration

if __name__ == "__main__":
    fig = plt.figure(figsize=(10,10))
    pd_phaseplot_boltzmann(fig)
    plt.show()

