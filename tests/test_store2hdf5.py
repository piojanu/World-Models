import os
import tempfile
import h5py
import numpy as np
import pytest

from unittest.mock import MagicMock
from humblerl import Transition
from humblerl.environments import Discrete, Continuous
from utils import StoreTransitions, convert_data_with_vae
from keras.models import Model


class TestStoreTransitions(object):

    LATENT_DIM = 16
    CHUNK_SIZE = 48
    STATE_SHAPE = (8, 8, 3)
    MIN_TRANSITIONS = 96
    MIN_EPISODES = 96
    N_TRANSITIONS = 1024
    N_GAMES = 64

    def setup(self):
        self.hdf5_file, self.hdf5_path = tempfile.mkstemp()

    def teardown(self):
        os.close(self.hdf5_file)
        os.remove(self.hdf5_path)

    @pytest.fixture()
    def vae_encoder(self):
        def vae_return(batch):
            return np.random.uniform(-1, 1, (3, batch.shape[0], self.LATENT_DIM))
        mock = MagicMock(spec=Model)
        mock.predict.side_effect = vae_return
        return mock

    def get_random_transition(self, action_space, is_terminal=False):
        return Transition(
            state=np.random.randint(0, 255, size=self.STATE_SHAPE, dtype=np.uint8),
            action=action_space.sample(),
            reward=np.random.normal(0, 1),
            next_state=np.random.randint(0, 255, size=self.STATE_SHAPE, dtype=np.uint8),
            is_terminal=is_terminal
        )

    def generate_transitions(self, action_space):
        callback = StoreTransitions(self.hdf5_path, self.STATE_SHAPE, action_space,
                                    self.MIN_TRANSITIONS, self.MIN_EPISODES, self.CHUNK_SIZE)
        transitions = []
        for idx in range(self.N_TRANSITIONS):
            transition = self.get_random_transition(
                action_space, is_terminal=(idx + 1) % 16 == 0)
            transitions.append(transition)
            callback.on_step_taken(idx, transition, None)
        callback.on_loop_end(False)
        return transitions

    def test_discrete_action_space(self):
        action_space = Discrete(3)

        transitions = self.generate_transitions(action_space)

        h5py_file = h5py.File(self.hdf5_path, "r")
        assert h5py_file.attrs["N_TRANSITIONS"] == self.N_TRANSITIONS
        assert h5py_file.attrs["N_GAMES"] == self.N_GAMES
        assert h5py_file.attrs["CHUNK_SIZE"] == self.CHUNK_SIZE
        assert np.all(h5py_file.attrs["STATE_SHAPE"] == self.STATE_SHAPE)
        assert h5py_file.attrs["ACTION_DIM"] == 1

        for idx, transition in enumerate(transitions):
            assert np.allclose(h5py_file['states'][idx], transition.state)
            assert h5py_file['actions'][idx][0] == transition.action
            assert np.allclose(h5py_file['rewards'][idx], transition.reward)

        for idx in range(self.N_GAMES + 1):
            assert h5py_file['episodes'][idx] == idx * 16

    def test_continous_action_space(self):
        action_space = Continuous(num=3, low=np.array([-1.0, 0.0, 0.0]),
                                  high=np.array([1.0, 1.0, 1.0]))

        transitions = self.generate_transitions(action_space)

        h5py_file = h5py.File(self.hdf5_path, "r")
        assert h5py_file.attrs["N_TRANSITIONS"] == self.N_TRANSITIONS
        assert h5py_file.attrs["N_GAMES"] == self.N_GAMES
        assert h5py_file.attrs["CHUNK_SIZE"] == self.CHUNK_SIZE
        assert np.all(h5py_file.attrs["STATE_SHAPE"] == self.STATE_SHAPE)
        assert h5py_file.attrs["ACTION_DIM"] == action_space.num

        for idx, transition in enumerate(transitions):
            assert np.allclose(h5py_file['states'][idx], transition.state)
            assert np.allclose(h5py_file['actions'][idx], transition.action)
            assert np.allclose(h5py_file['rewards'][idx], transition.reward)

        for idx in range(self.N_GAMES + 1):
            assert h5py_file['episodes'][idx] == idx * 16

    def test_convert_data_with_vae(self, vae_encoder):
        action_space = Discrete(3)

        transitions = self.generate_transitions(action_space)

        file_out, path_out = tempfile.mkstemp()

        convert_data_with_vae(vae_encoder, self.hdf5_path, path_out, self.LATENT_DIM)

        h5py_file = h5py.File(path_out, "r")
        assert h5py_file.attrs["N_TRANSITIONS"] == self.N_TRANSITIONS
        assert h5py_file.attrs["N_GAMES"] == self.N_GAMES
        assert h5py_file.attrs["CHUNK_SIZE"] == self.CHUNK_SIZE
        assert np.all(h5py_file.attrs["STATE_SHAPE"] == (2, self.LATENT_DIM))
        assert h5py_file.attrs["ACTION_DIM"] == 1

        for idx, transition in enumerate(transitions):
            assert np.all(h5py_file['states'][idx].shape == (2, self.LATENT_DIM))
            assert np.allclose(h5py_file['actions'][idx], transition.action)
            assert np.allclose(h5py_file['rewards'][idx], transition.reward)

        for idx in range(self.N_GAMES + 1):
            assert h5py_file['episodes'][idx] == idx * 16

        os.close(file_out)
        os.remove(path_out)
