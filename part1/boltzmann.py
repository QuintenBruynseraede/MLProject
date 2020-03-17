import pyspiel
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.egt import utils
from open_spiel.python.egt import visualization
import matplotlib.pyplot as plt
import numpy as np


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

def matrix_pd_phaseplot_boltzmann():
    game = pyspiel.load_game("matrix_pd")
    payoff_tensor = game_payoffs_array(game)
    A = payoff_tensor[0]
    B = payoff_tensor[1]
    x_vec = np.linspace(0,1,num=11)
    y_vec = np.linspace(0,1,num=11)

    for x in x_vec:
        for y in y_vec:
            x_policy = np.array([[x, 1-x]]).T
            y_policy = np.array([[y, 1-y]]).T
            dx = x * (np.matmul(A,y_policy)[0] - np.matmul(x_policy.T,np.matmul(A,y_policy)))
            dy = y * (np.matmul(x_policy.T,B).T[0] - np.matmul(x_policy.T,np.matmul(B,y_policy)))
            length = 1
            print("({},{}): dx={},dy={}".format(round(x,1),round(y,1),dx,dy))
            ax = plt.axes()
            scale = 0.05
            ax.arrow(x,y,dx[0][0]*scale,dy[0][0]*scale)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()

if __name__ == "__main__":
    matrix_pd_phaseplot_boltzmann()

