import pyspiel
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
import matplotlib.pyplot as plt
import numpy as np
import ternary
from ternary.helpers import project_point
import matrix_games as mg


#Constructs a phase plot for the prisoners dilemma
def matrix_pd_phaseplot(size, learning_rate):
    game = pyspiel.load_game("matrix_pd")
    payoff_matrix = game_payoffs_array(game)
    dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)
    print(payoff_matrix)
    print(dyn.dynamics)

    xy = np.mgrid[0:1.1:1 / size, 0:1.1:1 / size].reshape(2, -1).T
    for p in xy:
        r = dyn(p)
        res = p + learning_rate * r
        plt.arrow(p[0], p[1], (res - p)[0], (res - p)[1], head_width=0.01)
    
    plt.title("Phaseplot prisoners dilemma")
    plt.xticks([0, 1], ["Cooperate", "Defect"])
    plt.yticks([0, 1], ["Cooperate", "Defect"])
    plt.xlabel("Player 1")
    plt.ylabel("Player 2")
    plt.show()


#Construcs a phaseplot for the matching pennies problem
def matrix_mp_phaseplot(size, learning_rate):
    game = pyspiel.load_game("matrix_mp")
    payoff_matrix = game_payoffs_array(game)
    payoff_matrix = np.array([[[1,0],[0,1]],[[0,1],[1,0]]])
    print(payoff_matrix[0])
    print(payoff_matrix[1])

    dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)

    xy = np.mgrid[0:1.1:1 / size, 0:1.1:1 / size].reshape(2, -1).T
    for p in xy:
        res = p + learning_rate * dyn(p)
        plt.arrow(p[0], p[1], (res - p)[0], (res - p)[1], head_width=0.01)

    plt.title("Phase plot matching pennies")
    plt.xticks([0, 1], ["Heads", "Tails"])
    plt.yticks([0, 1], ["Heads", "Tails"])
    plt.xlabel("Player 1")
    plt.ylabel("Player 2")
    plt.show()


#Construcs a phaseplot for the battle of the sexes problem
def matrix_bots_phaseplot(size, learning_rate):
    payoff_matrix = np.array([[[3, 0], [0, 2]], [[2, 0], [0, 3]]])
    print(payoff_matrix)

    dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)

    xy = np.mgrid[0:1.1:1 / size, 0:1.1:1 / size].reshape(2, -1).T
    for p in xy:
        res = p + learning_rate * dyn(p)
        plt.arrow(p[0], p[1], (res - p)[0], (res - p)[1], head_width=0.01)

    plt.xticks([0, 1], ["Football", "Opera"])
    plt.yticks([0, 1], ["Football", "Opera"])
    plt.xlabel("Player 1")
    plt.ylabel("Player 2")
    plt.show()


#Construcs a phaseplot for a biased rock papers scissors
def matrix_rps_phaseplot(size, learning_rate):

    #Construct actions
    actions = mg.get_replicator_dynamics("matrix_rpsw",size,learning_rate)

    #Construct phaseplot
    arrows_every = 2
    _, tax = ternary.figure()
    tax.set_title("Phaseplot RPS", fontsize=15)
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
    # print(pyspiel.registered_names())
    # matrix_pd_phaseplot(20,0.01)
    # matrix_mp_phaseplot(20,0.01)
    matrix_bots_phaseplot(20,0.01)
    # matrix_rps_phaseplot(10000,0.05)
    #plt.plot(actions)
    plt.show()