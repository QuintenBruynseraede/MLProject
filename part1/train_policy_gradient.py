from open_spiel.python.algorithms.policy_gradient import PolicyGradient as pg
import tensorflow as tf
from open_spiel.python import rl_environment
import matplotlib.pyplot as plt
import pyspiel

game_name = "matrix_pd"
env = rl_environment.Environment(game_name)
num_actions = env.action_spec()["num_actions"]
policy_1 = [0.75,.25]
policy_2 = [0.4,0.6]

step_size = 0.1
disc = 0.8
eps = 0.99
agent1 = pg(tf.Session(),0,1,num_actions,pi_learning_rate=0.01, critic_learning_rate=0.01,batch_size=5,initial_policy=policy_1)
agent2 = pg(tf.Session(),1,1,num_actions,pi_learning_rate=0.01, critic_learning_rate=0.01,batch_size=5,initial_policy=policy_2)

iterations = 20000
p1 = list()
p2 = list()
for episode in range(iterations):
    time_step = env.reset()
    i = 0
    while not time_step.last():
        step1 = agent1.step(time_step)
        step2 = agent2.step(time_step)
        output = [step1.action, step2.action]
        probs = [step1.probs[0], step2.probs[0]]
        print(probs)
        time_step = env.step(output)
        # print("Episode: " + str(episode) + ", Iteration: " + str(i) + str(probs))

    p1.append(probs[0])
    p2.append(probs[1])
    agent1.step(time_step)
    agent2.step(time_step)

plt.plot(p1, p2)
plt.xlim((0,1))
plt.ylim((0,1))
plt.show()


