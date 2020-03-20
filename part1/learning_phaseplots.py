import numpy as np
from training import self_learn
import matplotlib.pyplot as plt
import phase_plots as pp


epsilon = 0.999
iterations = 10000
experiments = 10
step_size = 0.5

def qlearner():
    pp.matrix_pd_phaseplot()

    probs_list1 = np.zeros((experiments,iterations,2))
    probs_list2 = np.zeros((experiments,iterations,2))
    for i in range(experiments):
        probs1, probs2 = self_learn("matrix_pd", "cross_learner",iterations=10000,learning_rate=0.001,initial_policy=[[0.4,0.6],[0.7,0.3]])
        probs_list1[i,:,:] = probs1
        probs_list2[i,:,:] = probs2

    plt.show()
    

qlearner()
