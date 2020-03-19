from part1.learners.cross_learner import CrossLearner as cl
from open_spiel.python import rl_environment
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python.egt.utils import game_payoffs_array
import numpy as np


class CrossLearner:

    def __init__(self,num_actions, initial_policy, rewards):
        self._policy = initial_policy
        self._rewards = rewards
        self._num_actions = num_actions

    def step(self):
        action = np.random.choice(range(self._num_actions), p=np.array(self._policy))
        r = self._rewards
        for a in range(self._num_actions):
            if a == action:
                self._policy[a] += (r[a] - self._policy[a] * r[a])
            else:
                self._policy[a] += (-r[a] * self._policy[a])

        return {"action":action,"probs":self._policy}

game = pyspiel.load_game("matrix_pd")
env = rl_environment.Environment(game)
num_actions = env.action_spec()["num_actions"]
payoff_array = game_payoffs_array(game)
rewards1 = np.mean(np.array(payoff_array[0]),axis=0)
rewards2 = np.mean(np.array(payoff_array[1]),axis=1)

rewards1 = rewards1 / sum(rewards1)
rewards2 = rewards2 / sum(rewards2)

agent1 = CrossLearner(num_actions,[0.4,0.6],rewards1)
agent2 = CrossLearner(num_actions,[0.75,0.25],rewards2)

iterations = 10000
p1 = list()
p2 = list()
for episode in range(iterations):
    time_step = env.reset()
    i = 0
    step1 = agent1.step()
    step2 = agent2.step()
    output = [step1["action"], step2["action"]]
    probs = [step1["probs"][0], step2["probs"][0]]
    print(probs)

    p1.append(probs[0])
    p2.append(probs[1])

plt.plot(p1, p2)
plt.xlim((0,1))
plt.ylim((0,1))
plt.show()


