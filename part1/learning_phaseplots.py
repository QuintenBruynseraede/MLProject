from open_spiel.python.algorithms import tabular_qlearner
import numpy as np
from training import train_algorithms_qlearn_qlearn
import matplotlib.pyplot as plt
import phase_plots as pp


epsilon = 0.999
iterations = 330000
experiments = 1
step_size = 10

def qlearner():
    probs = [[0.5,0.75],[0.2,0.2],[0.8,0.15]]
    #probs = [[0.6,0.6]]
    xlist = list()
    ylist = list()
    for iprobs in probs:
        pp.matrix_mp_phaseplot()

        probs_list1 = np.zeros((experiments,iterations,2))
        probs_list2 = np.zeros((experiments,iterations,2))
        for i in range(experiments):
            probs1, probs2 = train_algorithms_qlearn_qlearn("matrix_mp", epsilon,0,iprobs, step_size,iterations)
            probs_list1[i,:,:] = probs1
            probs_list2[i,:,:] = probs2

        p1 = np.mean(probs_list1,0,dtype=np.float64)
        p2 = np.mean(probs_list2,0, dtype=np.float64)
        #print(p1[-10:-1])
        #print(p2[-10:-1])
        xlist.append(p1[:,0])
        ylist.append(p2[:,0])
        #plt.plot(p1[:,0],p2[:,0])
    
    plt.plot(xlist[0],ylist[0],xlist[1],ylist[1],xlist[2],ylist[2])
    #plt.plot(xlist[0],ylist[0])

    plt.show()
    

qlearner()
