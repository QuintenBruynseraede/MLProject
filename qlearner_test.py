import pyspiel
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent


'''
game.new_initial_state()
game.num_players()
state.is_terminal()
state.is_simultaneous_node()
state.is_chance_node()
state.legal_actions(int playerId)
state.apply_actions([a1,a2, ...] voor elke playerId)
state.returns()    | geeft result van het spel weer

'''

env = rl_environment.Environment("tic_tac_toe")
num_actions = env.action_spec()["num_actions"]
q_agent = tabular_qlearner.QLearner(0, num_actions)
ra2 = random_agent.RandomAgent(1, num_actions)
players = [q_agent, ra2]

for cur_episode in range(5000):
    if (cur_episode % 1000 == 0):
        print("cur_episode: "+str(cur_episode))

    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      agent_output = players[player_id].step(time_step)
      time_step = env.step([agent_output.action])

    for agent in players:
      agent.step(time_step)

#Test?
n = 100
win = 0
for i in range(n):
    time_step = env.reset()
    while not time_step.last():
        player_id = time_step.observations['current_player']
        agent_output = players[player_id].step(time_step)
        time_step = env.step([agent_output.action])
    for agent in players:
      agent.step(time_step)

    if time_step.rewards[0] > time_step.rewards[1]:
        win += 1


print(win/n)

