from open_spiel.python.algorithms import tabular_qlearner
import numpy as np
from training import train_algorithms_qlearn_qlearn
import matplotlib.pyplot as plt
import phase_plots as pp


epsilon = 0.999
iterations = 30000
experiments = 10
step_size = 0.5

def qlearner():
    probs = [0.25,0.5,0.75,]
    pp.matrix_pd_phaseplot()

    probs_list1 = np.zeros((experiments,iterations,2))
    probs_list2 = np.zeros((experiments,iterations,2))
    for i in range(experiments):
        probs1, probs2 = train_algorithms_qlearn_qlearn("matrix_pd", epsilon,0,"INITIAL", step_size,iterations)
        probs_list1[i,:,:] = probs1
        probs_list2[i,:,:] = probs2

    p1 = np.mean(probs_list1,0,dtype=np.float64)
    p2 = np.mean(probs_list2,0, dtype=np.float64)
    print(p1[-10:-1])
    print(p2[-10:-1])
    plt.plot(p1[:,0],p2[:,0])
    plt.show()

qlearner()