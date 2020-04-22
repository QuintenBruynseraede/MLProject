from open_spiel.python import rl_environment
from open_spiel.python.algorithms.dqn import DQN
from open_spiel.python.algorithms.tabular_qlearner import QLearner
from open_spiel.python.policy import TabularPolicy
from open_spiel.python.algorithms import get_all_states
import pyspiel
import tensorflow as tf


def self_train():
    env = rl_environment.Environment("kuhn_poker")
    num_actions = env.action_spec()["num_actions"]

    player1 = QLearner(0,num_actions)
    player2 = QLearner(1,num_actions)
    state_size = env.observation_spec()["info_state"][0]
    with tf.Session as sess:
        player1 = DQN(sess,0,11,state_representation_size=state_size,num_actions=num_actions)
        player1 = DQN(sess,1,11,state_representation_size=state_size,num_actions=num_actions)

    players = [player1, player2]


    iterations = 1000000
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

    game = pyspiel.load_game("kuhn_poker")
    all_states = get_all_states.get_all_states(
      game,
      depth_limit=-1,
      include_terminals=False,
      include_chance_states=False,
      to_string=lambda s: s.information_state_string())
    
    #Initialized to uniform for each state
    tabular_policy = TabularPolicy(game)


    for state in all_states:
        state_policy = tabular_policy.policy_for_key(state)
        print("State: {}, state_policy: {}".format(state,state_policy))


self_train()