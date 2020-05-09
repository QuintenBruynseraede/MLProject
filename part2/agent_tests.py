from absl import logging,flags,app
import tensorflow.compat.v1 as tf
from datetime import datetime
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_gradient
from open_spiel.python.algorithms import nfsp,dqn,cfr,neurd,rcfr, fictitious_play
from open_spiel.python.egt.examples import alpharank_example
from open_spiel.python.egt import alpharank
from agent_policies import NFSPPolicies,QLearnerPolicies,DQNPolicies,PolicyGradientPolicies, PolicyFromDict
from tournament import policy_to_csv
import matplotlib.pyplot as plt
import pyspiel
import pickle
from deepcfr_solver_modified import DeepCFRSolverModified
import numpy as np
import os

FLAGS = flags.FLAGS

flags.DEFINE_integer('episodes',int(1e5),"Number of training episodes")
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
    exploit_history = list()
    exploit_idx = list()

    tf.enable_eager_execution()
    game = pyspiel.load_game(FLAGS.game,
                           {"players": pyspiel.GameParameter(2)})
    agent_name = "cfr"
    cfr_solver = cfr.CFRSolver(game)
    checkpoint = datetime.now()
    for ep in range(FLAGS.episodes):
        cfr_solver.evaluate_and_update_policy()
        if ep % 100 == 0:
            delta = datetime.now() - checkpoint
            conv = exploitability.exploitability(game, cfr_solver.average_policy())
            exploit_idx.append(ep)
            exploit_history.append(conv)
            print("Iteration {} exploitability {} - {} seconds since last checkpoint".format(ep, conv,delta.seconds))
            checkpoint = datetime.now()
    
    pickle.dump([exploit_idx,exploit_history],open(FLAGS.game+"_"+agent_name+"_"+str(FLAGS.episodes)+".dat","wb"))

    now = datetime.now()
    policy = cfr_solver.average_policy()
    agent_name = "cfr"
    for pid in [1,2]:
        policy_to_csv(game, policy, f"policies/policy_"+now.strftime("%m-%d-%Y_%H-%M")+"_"+agent_name+"_"+str(pid+1)+"_+"+str(ep)+"episodes.csv")

def xfsp_train(_):
    exploit_history = list()
    exploit_idx = list()
    game = pyspiel.load_game(FLAGS.game,{"players": pyspiel.GameParameter(2)})
    fsp_solver = fictitious_play.XFPSolver(game)
    checkpoint = datetime.now()
    for ep in range(FLAGS.episodes):
        if (ep % 1000) == 0:
            delta = datetime.now() - checkpoint
            pol = policy.PolicyFromCallable(game, fsp_solver.average_policy_callable())
            conv = exploitability.exploitability(game,pol)
            exploit_history.append(conv)
            exploit_idx.append(ep)
            print("[XFSP] Iteration {} exploitability {} - {} seconds since last checkpoint".format(ep, conv,delta.seconds))
            checkpoint = datetime.now()


        fsp_solver.iteration()

    agent_name = "xfsp"
    pickle.dump([exploit_idx,exploit_history],open(FLAGS.game+"_"+agent_name+"_"+str(FLAGS.episodes)+".dat","wb"))

    pol = policy.PolicyFromCallable(game, fsp_solver.average_policy_callable())
    for pid in [1,2]:
        policy_to_csv(game, pol, f"policies/policy_"+now.strftime("%m-%d-%Y_%H-%M")+"_"+agent_name+"_"+str(pid+1)+"_+"+str(FLAGS.episodes)+"episodes.csv")

def rcfr_train(unused_arg):
    tf.enable_eager_execution()
    game = pyspiel.load_game(FLAGS.game,
            {"players": pyspiel.GameParameter(2)})
    models = [rcfr.DeepRcfrModel(
      game,
      num_hidden_layers=1,
      num_hidden_units=64 if FLAGS.game == "leduc_poker" else 13,
      num_hidden_factors=1,
      use_skip_connections=True) for _ in range(game.num_players())]
    patient = rcfr.RcfrSolver(
        game, models, False, True)
    exploit_history = list()
    exploit_idx = list()

    def _train(model, data):
        data = data.shuffle(1000)
        data = data.batch(12)
        #data = data.repeat(1)  
        optimizer = tf.keras.optimizers.Adam(lr=0.005, amsgrad=True)    
        for x, y in data:
            optimizer.minimize(
                lambda: tf.losses.huber_loss(y, model(x)),  # pylint: disable=cell-var-from-loop
                model.trainable_variables)

    agent_name = "rcfr"
    checkpoint = datetime.now()
    for iteration in range(FLAGS.episodes):
        if (iteration % 100) == 0:
            delta = datetime.now() - checkpoint
            conv = pyspiel.exploitability(game, patient.average_policy())
            exploit_idx.append(iteration)
            exploit_history.append(conv)
            print("[RCFR] Iteration {} exploitability {} - {} seconds since last checkpoint".format(iteration, conv,delta.seconds))
            checkpoint = datetime.now()
        patient.evaluate_and_update_policy(_train)


    pickle.dump([exploit_idx,exploit_history],open(FLAGS.game+"_"+agent_name+"_"+str(FLAGS.episodes)+".dat","wb"))

    now = datetime.now()
    policy = patient.average_policy()

    for pid in [1,2]:
        policy_to_csv(game, policy, f"policies/policy_"+now.strftime("%m-%d-%Y_%H-%M")+"_"+agent_name+"_"+str(pid+1)+"_+"+str(FLAGS.episodes)+"episodes.csv")

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
    for ep in range(FLAGS.episodes):
        solver.evaluate_and_update_policy(_train)
        if ep % 100 == 0:
            conv = pyspiel.exploitability(game, solver.average_policy())
            exploit_history.append(conv)
            print("Iteration {} exploitability {}".format(ep, conv))
        
    now = datetime.now()
    policy = solver.average_policy()
    agent_name = "neurd"
    for pid in [1,2]:
        policy_to_csv(game, policy, f"policies/policy_"+now.strftime("%m-%d-%Y_%H-%M")+"_"+agent_name+"_"+str(pid+1)+"_+"+str(ep)+"episodes.csv")

    plt.plot([i for i in range(len(exploit_history))],exploit_history)
    plt.ylim(0.01,1)
    plt.yticks([1,0.1,0.01])
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


def train_dcfr(_):
  np.random.seed(0)
  game = pyspiel.load_game(FLAGS.game)

  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

  num_traversals = 5
  with tf.Session() as sess:
    deep_cfr_solver = DeepCFRSolverModified(
        sess,
        game,
        policy_network_layers=(64, 64) if FLAGS.game == "leduc_poker" else (32,32),
        advantage_network_layers=(32, 32) if FLAGS.game == "leduc_poker" else (16,16),
        num_iterations= FLAGS.episodes,
        num_traversals=num_traversals,
        learning_rate=1e-3,
        batch_size_advantage=int(1e5),
        batch_size_strategy=int(1e5),
        memory_capacity=int(5e5),
        eval_frequency = 100)
    sess.run(tf.global_variables_initializer())
    deep_cfr_solver.solve()

    exploit_idx, exploit_history = deep_cfr_solver.get_exploitabilities_from_memories(sess)
    agent_name = "dcfr"
    pickle.dump([exploit_idx,exploit_history],open(FLAGS.game+"_"+agent_name+"_"+str(FLAGS.episodes)+".dat","wb"))

    plt.plot(exploit_idx, exploit_history)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


def run_agents(sess, env, agents, expl_policies_avg):
    agent_name = "nfsp"
    write_policy_at = [1e4,1e5,1e6,3e6,5e6]
    sess.run(tf.global_variables_initializer())
    exploit_idx = list()
    exploit_history = list()
    for ep in range(FLAGS.episodes):
        if (ep + 1) % 10000 == 0:
            expl = exploitability.exploitability(env.game, expl_policies_avg)
            exploit_idx.append(ep)
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
            player_id = time_stcfr_trainep.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)


    pickle.dump([exploit_idx,exploit_history],open(FLAGS.game+"_"+agent_name+"_"+str(FLAGS.episodes)+".dat","wb"))

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
    app.run(train_dcfr)
