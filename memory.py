import numpy as np
import os.path

from third_party.humblerl import Callback


class StoreTrajectories2npz(Callback):
    """Save state, action, next_state trajectories with length of each into numpy archive.

    NOTE:
        For now it's coded with continuous state and discrete action spaces in mind!
    """

    def __init__(self, path):
        """Initialize trajectories saver.

        Args:
            path (str): Where to save numpy archive.
        """

        self.path = path

        self._states = []
        self._actions = []
        self._episod_lengths = []

    def on_episode_start(self, train_mode):
        self._states.append([])
        self._actions.append([])
        self._episod_lengths.append(0)

    def on_step_taken(self, transition, info):
        self._states[-1].append(transition.state)
        self._actions[-1].append(transition.action)
        self._episod_lengths[-1] += 1

    def on_loop_finish(self, is_aborted):
        longest_episode = max(self._episod_lengths)
        num_episodes = len(self._episod_lengths)

        states = np.zeros((num_episodes, longest_episode, len(self._states[0][0])))
        actions = np.zeros((num_episodes, longest_episode, 1))
        lengths = np.zeros((num_episodes, 1))

        for idx, (states_seq, actions_seq, length) in enumerate(zip(
                self._states, self._actions, self._episod_lengths)):
            states[idx, :length, :] = states_seq
            actions[idx, :length, 0] = actions_seq
            lengths[idx] = length

        np.savez(os.path.splitext(self.path)[0],  # Strip extension off path (if there is one)
                 states=states, actions=actions, lengths=lengths)
