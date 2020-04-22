from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
import os
from open_spiel.python.algorithms import policy_gradient
from datetime import datetime
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_gradient
from open_spiel.python.algorithms.dqn import DQN
from open_spiel.python.algorithms.nfsp import NFSP
from agent_policies import AgentPolicies
from tournament import policy_to_csv
import matplotlib.pyplot as plt

def dqn(game):
    env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    sess = tf.Session()
    player1 = DQN(sess,0,state_representation_size=info_state_size,num_actions=num_actions,learning_rate=0.2)
    player2 = DQN(sess,1,state_representation_size=info_state_size,num_actions=num_actions,learning_rate=0.2)
    players = [player1, player2]
    run_agents(game,players,sess)

def nfsp(game):
    env = rl_environment.Environment(game)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    
    sess =  tf.Session()
    players = [NFSP(sess,idx,state_representation_size=state_size,num_actions=num_actions,
                    hidden_layers_sizes=[64],
                    reservoir_buffer_capacity=20000000,
                    rl_learning_rate=0.1,
                    sl_learning_rate=0.005,
                    anticipatory_param=0.1,
                    batch_size=128,
                    learn_every=64) for idx in range(2)]  
    run_agents(game,players,sess)

def pgrad(game):
    sess = tf.Session()
    env = rl_environment.Environment(game)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    
    agents = [
        policy_gradient.PolicyGradient(
            sess,
            idx,
            state_size,
            num_actions,
            loss_str="rpg",
            hidden_layers_sizes=(128,)) for idx in range(2)]
    run_agents(game,agents,sess)

def run_agents(game,agents,sess):
    num_players = 2

    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    

    expl_policies_avg = AgentPolicies(env,agents)

    sess.run(tf.global_variables_initializer())
    num_episodes = 1000000
    exploit_history = list()
    for ep in range(num_episodes):
        if (ep + 1) % 1000 == 0:
            expl = exploitability.exploitability(env.game, expl_policies_avg)
            exploit_history.append(expl)
        if (ep + 1) % 10000 == 0:
            losses = [agent.loss for agent in agents]
            msg = "-" * 80 + "\n"
            msg += "{}: {}\n{}\n".format(ep + 1, expl, losses)
            logging.info("%s", msg)
            print(msg)

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)

    now = datetime.now()
    agent_name = "nfsp"
    for pid, agent in enumerate(agents):
        policy_to_csv(env.game, expl_policies_avg, f"policies/test_p_"+now.strftime("%m-%d-%Y_%H-%M")+"_"+agent_name+"_"+str(pid+1)+".csv")

    plt.plot([i for i in range(len(exploit_history))],exploit_history)
    plt.show()



if __name__ == "__main__":
    nfsp("leduc_poker")