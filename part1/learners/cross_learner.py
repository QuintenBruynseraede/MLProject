from open_spiel.python import rl_agent
import numpy as np

class CrossLearner(rl_agent.AbstractAgent):

    def __init__(self, player_id, num_actions,initial_policy,learning_rate=0.1, **agent_specific_kwargs):

        assert len(initial_policy) == num_actions
        assert sum(initial_policy) == 1

        self._player_id = player_id
        self._policy = list(initial_policy)
        self._num_actions = num_actions
        self._prev_info_state = None
        self._prev_action = None
        self._learning_rate = learning_rate

    def step(self, time_step, is_evaluation=False):

        info_state = str(time_step.observations["info_state"][self._player_id])

        action, probs,reward = None,None,None
        if not time_step.last():
            #Fix errors in accuracy
            policy = np.array(self._policy)
            policy[policy <= 0] = 0
            policy /= sum(policy)
            action = np.random.choice(range(self._num_actions), p=policy)
            probs = self._policy


        if self._prev_info_state and not is_evaluation:
            probs = self._policy
            reward = time_step.rewards[self._player_id]

            action = np.random.choice(range(self._num_actions),p=probs)
            for a in range(self._num_actions):
                if a == self._prev_action:
                    self._policy[a] = probs[a] + self._learning_rate * (reward - probs[a] * reward)
                else:
                    self._policy[a] = probs[a] + self._learning_rate * (-reward * probs[a])
            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                return

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action

        return rl_agent.StepOutput(action=action, probs=probs)

        pass