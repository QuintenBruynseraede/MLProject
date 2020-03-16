from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.algorithms import random_agent
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
import numpy as np
import collections
from matplotlib import pyplot as plt


def train_algorithms(algo1, algo2, game, iterations=10000):
    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]
    agent1 = algo1(0, num_actions)
    agent2 = algo2(1, num_actions)
    actions = np.zeros((iterations, 2))

    #Time-dependent list of actions probabilities throughout training
    #One list for each agent
    probs1 = np.zeros((iterations,num_actions))
    probs2 = np.zeros((iterations,num_actions))

    for episode in range(iterations):
        if episode % 1000 == 0:
            print("cur_episode: " + str(episode))

        time_step = env.reset()
        while not time_step.last():
            output = list()
            step1 = agent1.step(time_step)
            probs1[episode] = step1.probs
            #print("Agent 1: {} - {}".format(step1.action,step1.probs))
            output.append(step1.action)
            step2 = agent2.step(time_step)
            probs2[episode] = step2.probs
            #print("Agent 2: {} - {}".format(step2.action,step2.probs))
            output.append(step2.action)
            time_step = env.step(output)

        actions[episode] = output
        agent1.step(time_step)
        agent2.step(time_step)

    
    return probs1,probs2

iterations = 40
algo1 = tabular_qlearner.QLearner
algo2 = tabular_qlearner.QLearner
#algo2 = random_agent.RandomAgent

probs1,probs2 = train_algorithms(algo1, algo2, "matrix_pd",iterations=iterations)
x = range(0,iterations)

plt.title("Odds of cooperation for both agents: expected to converge to zero.")
plt.plot(x,probs1[:,0], label="Agent 1")
plt.plot(x,probs2[:,0], label="Agent 2")
plt.legend()


plt.show()
