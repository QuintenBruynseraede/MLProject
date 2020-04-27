from absl import logging,flags,app
import tensorflow.compat.v1 as tf
from datetime import datetime
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_gradient
from open_spiel.python.algorithms import nfsp,dqn,cfr,neurd
from agent_policies import NFSPPolicies,QLearnerPolicies,DQNPolicies,PolicyGradientPolicies
from tournament import policy_to_csv
import matplotlib.pyplot as plt
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer('episodes',int(5e6+10),"Number of training episodes")
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

def cfr_train(unused_arg):
    tf.enable_eager_execution()
    game = pyspiel.load_game(FLAGS.game,
                           {"players": pyspiel.GameParameter(2)})
    cfr_solver = cfr.CFRSolver(game)
    for i in range(FLAGS.episodes):
        cfr_solver.evaluate_and_update_policy()
        if i % 100 == 0:
            conv = exploitability.exploitability(game, cfr_solver.average_policy())
            print("Iteration {} exploitability {}".format(i, conv))
    
    now = datetime.now()
    policy = cfr_solver.average_policy()
    agent_name = "cfr"
    for pid in [1,2]:
        policy_to_csv(game, policy, f"policies/test_p_"+now.strftime("%m-%d-%Y_%H-%M")+"_"+agent_name+"_"+str(pid+1)+".csv")


def neurd_train(unudes_arg):
    tf.enable_eager_execution()

    game = pyspiel.load_game(FLAGS.game,
                           {"players": pyspiel.GameParameter(2)})

    models = []
    for _ in range(game.num_players()):
        models.append(
            neurd.DeepNeurdModel(
                game,
                num_hidden_layers=1,
                num_hidden_units=13,
                num_hidden_factors=8,
                use_skip_connections=True,
                autoencode=False))
    solver = neurd.CounterfactualNeurdSolver(game, models)

    def _train(model, data):
        neurd.train(
            model,
            data,
            batch_size=100,
            step_size=1,
            threshold=2,
            autoencoder_loss=(None))

    exploit_history = list()
    for i in range(FLAGS.episodes):
        solver.evaluate_and_update_policy(_train)
        if i % 100 == 0:
            conv = pyspiel.exploitability(game, solver.average_policy())
            exploit_history.append(conv)
            print("Iteration {} exploitability {}".format(i, conv))
        
    now = datetime.now()
    policy = PolicyFromDict(solver.current_policy())
    agent_name = "neurd"
    for pid, agent in enumerate(models):
        policy_to_csv(game, policy, f"policies/test_p_"+now.strftime("%m-%d-%Y_%H-%M")+"_"+agent_name+"_"+str(pid+1)+".csv")

    plt.plot([i for i in range(len(exploit_history))],exploit_history)
    plt.ylim(0.01,1)
    plt.yticks([1,0.1,0.01])
    plt.yscale("log")
    plt.xscale("log")
    plt.show()




def run_agents(sess, env, agents, expl_policies_avg):
    agent_name = "nfsp"
    write_policy_at = [1e4,1e5,1e6,3e6,5e6]
    sess.run(tf.global_variables_initializer())
    exploit_history = list()
    for ep in range(FLAGS.episodes):
        if (ep + 1) % 10000 == 0:
            expl = exploitability.exploitability(env.game, expl_policies_avg)
            exploit_history.append(expl)
            with open("exploitabilities.txt","a") as f:
                f.write(str(expl)+"\n")
            losses = [agent.loss for agent in agents]
            msg = "-" * 80 + "\n"
            msg += "{}: {}\n{}\n".format(ep + 1, expl, losses)
            logging.info("%s", msg)

        if ep in write_policy_at:
            for pid, agent in enumerate(agents):
                policy_to_csv(env.game, expl_policies_avg, f"policies/policy_"+agent_name+"_"+datetime.now().strftime("%m-%d-%Y_%H-%M")+"_"+str(pid+1)+"_"+str(ep)+"episodes.csv")    

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
    for pid, agent in enumerate(agents):
        policy_to_csv(env.game, expl_policies_avg, f"policies/policy_"+now.strftime("%m-%d-%Y_%H-%M")+"_"+agent_name+"_"+str(pid+1)+"_+"+str(ep)+"episodes.csv")

   
    plt.plot([i for i in range(len(exploit_history))],exploit_history)
    plt.ylim(0.01,1)
    plt.yticks([1,0.1,0.01])
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    app.run(nfsp_train)
