from absl import logging,flags,app
import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms import policy_gradient
from datetime import datetime
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_gradient
from open_spiel.python.algorithms import nfsp,dqn
from agent_policies import NFSPPolicies,QLearnerPolicies,DQNPolicies,PolicyGradientPolicies
from tournament import policy_to_csv
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_integer('episodes',int(5e5),"Number of training episodes")
flags.DEFINE_string('game',"kuhn_poker","Game to be played by the agents")


def dqn_train(unused_arg):
    env = rl_environment.Environment(FLAGS.game)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    sess =  tf.Session()
    players = [dqn.DQN(sess,idx,state_representation_size=state_size,num_actions=num_actions,
                    hidden_layers_sizes=[64],
                    reservoir_buffer_capacity=2e6,
                    batch_size=128,
                    learn_every=64,
                    replay_buffer_capacity=2e5,
                    epsilon_decay_duration=FLAGS.episodes,
                    epsilon_start=0.06,
                    eplsilon_end=0.001) for idx in range(2)]  
    expl_policies_avg = NFSPPolicies(env,players,nfsp.MODE.average_policy)
    run_agents(sess,env,players,expl_policies_avg)
    sess.close()

def nfsp_train(unused_arg):
    env = rl_environment.Environment(FLAGS.game)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    kwargs = {
      "replay_buffer_capacity": 2e5,
      "epsilon_decay_duration": FLAGS.episodes,
      "epsilon_start": 0.06,
      "epsilon_end": 0.001,
  }

    sess =  tf.Session()
    players = [nfsp.NFSP(sess,idx,state_representation_size=state_size,num_actions=num_actions,
                    hidden_layers_sizes=[64],
                    reservoir_buffer_capacity=2e6,
                    rl_learning_rate=0.1,
                    sl_learning_rate=0.005,
                    anticipatory_param=0.1,
                    batch_size=128,
                    learn_every=64,**kwargs) for idx in range(2)]  
    expl_policies_avg = NFSPPolicies(env,players,nfsp.MODE.average_policy)
    run_agents(sess,env,players,expl_policies_avg)
    sess.close()

def pgrad_train(unused_arg):
    sess = tf.Session()
    env = rl_environment.Environment(FLAGS.game)
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
    expl_policies_avg = PolicyGradientPolicies(env,agents)
    run_agents(sess,env,agents,expl_policies_avg)

def run_agents(sess, env, agents, expl_policies_avg):
    
    sess.run(tf.global_variables_initializer())
    exploit_history = list()
    for ep in range(FLAGS.episodes):
        if (ep + 1) % 1000 == 0:
            expl = exploitability.exploitability(env.game, expl_policies_avg)
            exploit_history.append(expl)
        if (ep + 1) % 1000 == 0:
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
    plt.ylim(0.01,1)
    plt.yticks([1,0.1,0.01])
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    app.run(nfsp_train)
