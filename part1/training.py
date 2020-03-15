from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.algorithms import random_agent
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
import numpy as np
from matplotlib import pyplot as plt


def train_algorithms(algo1, algo2, game, iterations=10000):
    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]
    agent1 = algo1(0, num_actions)
    agent2 = algo2(1, num_actions)
    actions = np.zeros((iterations, 2))

    for episode in range(iterations):
        if episode % 1000 == 0:
            print("cur_episode: " + str(episode))

        time_step = env.reset()
        while not time_step.last():
            output = list()
            output.append(agent1.step(time_step).action)
            output.append(agent2.step(time_step).action)
            time_step = env.step(output)

        actions[episode] = output
        agent1.step(time_step)
        agent2.step(time_step)
    return actions


algo1 = tabular_qlearner.QLearner
algo2 = tabular_qlearner.QLearner
actions = train_algorithms(algo1, algo2, "matrix_pd")
plt.plot(actions[:, 0])
plt.show()
