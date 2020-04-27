from open_spiel.python import rl_agent
import numpy as np

class DeterministicAgent(rl_agent.AbstractAgent):
    
    _preferred_action = 0

    def __init__(self, player_id, num_actions,preferred_action,name="random_agent"):
        assert num_actions > 0
        self._player_id = player_id
        self._num_actions = num_actions
        self._preferred_action = preferred_action

    def step(self, time_step, is_evaluation=False):
        # If it is the end of the episode, don't select an action.
        if time_step.last():
            return

        # Pick a random legal action.
        cur_legal_actions = time_step.observations["legal_actions"][self._player_id]
        if self._preferred_action in cur_legal_actions:
            probs = np.zeros(self._num_actions)
            probs[self._preferred_action] = 1
            return rl_agent.StepOutput(action=self._preferred_action, probs=probs)
        else:
            action = np.random.choice(cur_legal_actions)
            probs = np.zeros(self._num_actions)
            probs[cur_legal_actions] = 1.0 / len(cur_legal_actions)

        return rl_agent.StepOutput(action=action, probs=probs)
