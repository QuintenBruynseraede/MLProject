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

    sub.set_title("Phaseplot Matching pennies")
    sub.set_xlabel("Player 1")
    sub.set_ylabel("Player 2")
    return sub


def pd_phaseplot_boltzmann(fig):
    #Load game and payoff matrices
    size = 111
    game = pyspiel.load_game("matrix_mp")
    #A = game_payoffs_array(game)[0]
    #print(A)
    payoff_tensor = np.array([[[3, 0],[0, 2]],[[2, 0],[0,3]]])

    #Make phase plot using LFAQ
    dyn = dynamics.MultiPopulationDynamics(payoff_tensor, lenient_boltzmannq)
    sub = fig.add_subplot(1,2,1, projection="2x2")
    sub.quiver(dyn)

    startx = [0.75, 0.45,0.85,0.3]
    starty = [0.75, 0.84,0.45,0.7]

    #Follow trajectory for 5 points
    for i in range(4):
        p_x = startx[i]
        p_y = starty[i]
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
    sub.set_title("τ = 50, κ = 1")

    return fig

def lenient_boltzmannq(state, fitness):
    temperature = 50
    kappa = 1
    
    #PD
    #A = np.array([[ 5, 0],[10, 1]])
    #MP
    A = np.array([[ 5, 0],[10, 1]])
    #RPS
    #A = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
    x = list()
    y = list()
    x.append(fitness[0])
    x.append(1-fitness[0])
    y.append(fitness[1])
    y.append(1-fitness[1])

    fitness_exploitation = list()
    for i in range(len(fitness)):
        term = 0
        for j in range(len(fitness)):
            term += A[i][j] * y[j] * ( math.pow(sum([y[k] for k in range(2) if A[i][k] <= A[i][j]]),kappa) - math.pow(sum([y[k] for k in range(2) if A[i][k] < A[i][j]]),kappa)) / sum([y[k] for k in range(2) if A[i][k] == A[i][j]])
        #j=1
        #term1 = A[i][1] * y[1] * ( math.pow(sum([y[k] for k in range(2) if A[i][k] <= A[i][1]]),kappa) - math.pow(sum([y[k] for k in range(2) if A[i][k] < A[i][1]]),kappa)) / sum([y[k] for k in range(2) if A[i][k] == A[i][1]])
        fitness_exploitation.append(term)
    
    exploitation = (1. / temperature) * dynamics.replicator(state, fitness_exploitation)
    exploration = (np.log(state) - state.dot(np.log(state).transpose()))
    return exploitation - state * exploration

def rps_phaseplot_boltzmann(fig):
    size = 111

    game = pyspiel.load_game("matrix_rps")
    payoff_tensor = game_payoffs_array(game)
    print(payoff_tensor[0])
    dyn = dynamics.SinglePopulationDynamics(payoff_tensor, lenient_boltzmannq)
    sub = fig.add_subplot(size, projection="3x3")
    sub.quiver(dyn)

    sub.set_title("Phaseplot Rock Paper Scissors")
    return sub

if __name__ == "__main__":
    fig = plt.figure(figsize=(10,10))
    pd_phaseplot_boltzmann(fig)
    #rps_phaseplot_boltzmann(fig)
    plt.show()

