from learners.cross_learner import CrossLearner as cl
from open_spiel.python.algorithms.tabular_qlearner import QLearner as ql
from open_spiel.python.algorithms import random_agent
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
import numpy as np
import collections
from matplotlib import pyplot as plt

def self_learn(game, algorithm_name, iterations = 10000, **kwargs):
    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]
    kwargs["num_actions"] = num_actions
    agent1, agent2 = get_agents(algorithm_name,kwargs=kwargs)

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
            output.append(step1.action)
            step2 = agent2.step(time_step)
            probs2[episode] = step2.probs
            output.append(step2.action)
            time_step = env.step(output)

        actions[episode] = output
        agent1.step(time_step)
        agent2.step(time_step)

    return probs1,probs2


def get_agents(algorithm_name, kwargs):
    agent1, agent2 = None,None
    if algorithm_name == "q_learner":
        num_actions = kwargs.get("num_actions")
        epsilon = kwargs.get("epsilon",0.2)
        discount_factor = kwargs.get("discount_factor",1)
        step_size = kwargs.get("step_size",0.5)
        agent1 = ql(0,num_actions=num_actions,step_size=step_size,epsilon=epsilon,discount_factor=discount_factor)
        agent2 = ql(1,num_actions=num_actions,step_size=step_size,epsilon=epsilon,discount_factor=discount_factor)

    elif algorithm_name == "cross_learner":
        num_actions = kwargs.get("num_actions")
        initial_policy = kwargs.get("initial_policy",[[1/num_actions for _ in range(num_actions)],[1/num_actions for _ in range(num_actions)]])
        learning_rate = kwargs.get("learning_rate",0.01)
        agent1 = cl(0,num_actions,initial_policy[0],learning_rate)
        agent2 = cl(1,num_actions,initial_policy[1],learning_rate)

    return agent1,agent2