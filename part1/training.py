# from open_spiel.python.algorithms.tabular_qlearner import QLearnerInit as qli
from learners.cross_learner import CrossLearner as cl
from open_spiel.python.algorithms.tabular_qlearner import QLearner as ql
from open_spiel.python.algorithms import random_agent
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
import numpy as np
import collections
from matplotlib import pyplot as plt

def train_algorithms_qlearn_qlearn(game, epsilon, discount_factor, initial_probs,step_size,iterations=10000):
    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]
    # p1 = [initial_probs[0], 1-initial_probs[0]]
    # p2 = [initial_probs[1], 1-initial_probs[1]]
    # agent1 = qli(0, num_actions, epsilon=epsilon, discount_factor=discount_factor,initial_probs=p2,step_size=step_size)
    # agent2 = qli(1, num_actions, epsilon=epsilon, discount_factor=discount_factor,initial_probs=p2,step_size=step_size)
    agent1 = cl(0, num_actions,[0.4,0.6],learning_rate=0.0001)
    agent2 = cl(1, num_actions, [0.75,0.25],learning_rate=0.0001)

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