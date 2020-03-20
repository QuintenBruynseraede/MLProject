import numpy as np
from training import self_learn
import matplotlib.pyplot as plt
import phase_plots as pp
import ternary

iterations = 30000
experiments = 5

def qlearner():
    _,game = pp.matrix_rps_biased_phaseplot()

    policies = [(0.6,0.4),(0.5,0.75),(0.8,0.5)] #for non_ternary plots
    policies = [(0.1,0.5,0.4),(0.6,0.2,0.2)] #rps
    for probs in policies:
        probs_list1 = np.zeros((experiments,iterations,3)) #Change to 3 for rps
        probs_list2 = np.zeros((experiments,iterations,3)) #Change to 3 for rps
        for i in range(experiments):
            if len(probs) == 2:
                policy = [(probs[0], 1-probs[0]),
                          (probs[1], 1-probs[1])]
            else:
                policy = [probs,probs]
            print(policy)
            probs1, probs2 = self_learn(game, "cross_learner",iterations=iterations,learning_rate=0.001,initial_policy=policy)
            probs_list1[i,:,:] = probs1
            probs_list2[i,:,:] = probs2

        p1 = np.mean(probs_list1,axis=0)
        p2 = np.mean(probs_list2,axis=0)

        # plt.plot(p1[:,0],p2[:,0])     #Non_ternary plots (all matrix games but rps)

        #rps:
        plt.plot(p1)

    plt.show()
    

qlearner()
