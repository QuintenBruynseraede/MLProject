from open_spiel.python import rl_environment
from open_spiel.python.algorithms.tabular_qlearner import QLearner
from open_spiel.python.policy import PolicyFromCallable


def self_train():
    env = rl_environment.Environment("kuhn_poker")
    num_actions = env.action_spec()["num_actions"]

    player1 = QLearner(0,num_actions)
    player2 = QLearner(1,num_actions)
    players = [player1, player2]


    iterations = 10000
    for episode in range(iterations):
        if episode % 1000 == 0:
            print("Curr_episode", str(episode))

        time_step = env.reset()
        while not time_step.last():
            curr_player_id = time_step.current_player()
            agent_output = players[curr_player_id].step(time_step)
            time_step = env.step([agent_output.action])

        for player in players:
            player.step(time_step)


    print(player1._q_values)

self_train()