import logging as log


import numpy as np
import os.path
import torch
import torch.nn as nn
import torch.optim as optim

from third_party.humblerl import Callback, Vision
from third_party.torchtrainer import TorchTrainer
from torch.distributions import Normal
from torch.utils.data import Dataset

from utils import get_model_path_if_exists


class MDNVision(Vision, Callback):
    def __init__(self, vae_model, mdn_model, latent_dim, state_processor_fn):
        """Initialize vision processors.

        Args:
            vae_model (keras.Model): Keras VAE encoder.
            mdn_model (torch.nn.Module): PyTorch MDN-RNN memory.
            latent_dim (int): Latent space dimensionality.
            state_processor_fn (function): Function for state processing. It should
                take raw environment state as an input and return processed state.

        Note:
            In order to work, this Vision system must be also passed as callback to 'hrl.loop(...)'!
        """

        self.vae_model = vae_model
        self.mdn_model = mdn_model
        self.latent_dim = latent_dim
        self.state_processor_fn = state_processor_fn

    def __call__(self, state, reward=0.):
        return self.process_state(state), reward

    def process_state(self, state):
        # NOTE: [0][0] <- it gets first in the batch latent space mean (mu)
        latent = self.vae_model.predict(self.state_processor_fn(state)[np.newaxis, :])[0][0]
        memory = self.mdn_model.hidden[0].cpu().detach().numpy()

        return np.concatenate((latent, memory.flatten()))

    def on_episode_start(self, episode, train_mode):
        self.mdn_model.init_hidden(1)

    def on_step_taken(self, step, transition, info):
        state = torch.from_numpy(transition.state[:self.latent_dim]).view(1, 1, -1)
        action = torch.from_numpy(np.array([transition.action])).view(1, 1, -1)
        self.mdn_model(state, action)


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

    def on_episode_start(self, episode, train_mode):
        self._states.append([])
        self._actions.append([])
        self._episod_lengths.append(0)

    def on_step_taken(self, step, transition, info):
        self._states[-1].append(transition.state)
        self._actions[-1].append(transition.action)
        self._episod_lengths[-1] += 1

    def on_loop_end(self, is_aborted):
        longest_episode = max(self._episod_lengths)
        num_episodes = len(self._episod_lengths)

        states = np.zeros((num_episodes, longest_episode, *self._states[0][0].shape))
        actions = np.zeros((num_episodes, longest_episode, 1))
        lengths = np.zeros((num_episodes, 1))

        for idx, (states_seq, actions_seq, length) in enumerate(zip(
                self._states, self._actions, self._episod_lengths)):
            states[idx, :length, :] = states_seq
            actions[idx, :length, 0] = actions_seq
            lengths[idx] = length

        np.savez(os.path.splitext(self.path)[0],  # Strip extension off path (if there is one)
                 states=states, actions=actions, lengths=lengths)


class MDNDataset(Dataset):
    """Dataset of sequential data to train MDN-RNN."""

    def __init__(self, states, actions, lengths, sequence_len):
        """Initialize MDNDataset.

        Args:
            states (np.ndarray): Array of env states with shape 'N x S x *' where 'N' is number of
                examples, 'S' is max sequence length and '*' indicates any number of dimensions.
            actions (np.ndarray): Array of actions numbers with shape 'N x S x 1' where 'N' is
                number of examples, 'S' is max sequence length.
            lengths (np.ndarray): Array of sequences true lengths with shape 'N x 1' where 'N' is
                number of examples.
            sequence_len (int): Desired output sequence len.

        Note:
            Arrays should have the same size of the first dimension and their type should be the
            same as desired Tensor type.
        """

        self.states = torch.from_numpy(states)
        self.actions = torch.from_numpy(actions)
        self.lengths = torch.from_numpy(lengths)
        self.sequence_len = sequence_len

    def __getitem__(self, idx):
        """Get sequence at random starting position of given sequence length from episode `idx`."""
        offset = 1

        states = torch.zeros(self.sequence_len, self.states.shape[3], dtype=self.states.dtype)
        next_states = torch.zeros(self.sequence_len, self.states.shape[3], dtype=self.states.dtype)
        actions = torch.zeros(self.sequence_len, 1, dtype=self.actions.dtype)

        length = self.lengths[idx]
        # Sample where to start sequence of length `self.sequence_len` in episode `idx`
        # '- offset' because "next states" are offset by 'offset'
        start = np.random.randint(length - self.sequence_len - offset)

        # Sample latent states (this is done to prevent overfitting MDN-RNN to a specific 'z'.)
        latent = Normal(
            loc=self.states[idx, start:start + self.sequence_len + offset, 0],
            scale=self.states[idx, start:start + self.sequence_len + offset, 1]
        )
        z_samples = latent.sample()

        states = z_samples[:-offset]
        next_states = z_samples[offset:]
        actions = self.actions[idx, start:start + self.sequence_len]

        return [states, actions], [next_states]

    def __len__(self):
        return self.states.shape[0]


class MDN(nn.Module):
    def __init__(self, hidden_units, latent_dim, action_size, temperature, n_gaussians, num_layers=1):
        super(MDN, self).__init__()

        self.hidden_units = hidden_units
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.n_gaussians = n_gaussians
        self.num_layers = num_layers

        self.embedding = nn.Embedding.from_pretrained(torch.eye(int(action_size)))
        self.lstm = nn.LSTM(input_size=(latent_dim + action_size),
                            hidden_size=hidden_units,
                            num_layers=num_layers,
                            batch_first=True)
        self.pi = nn.Linear(hidden_units, n_gaussians * latent_dim)
        self.mu = nn.Linear(hidden_units, n_gaussians * latent_dim)
        self.logsigma = nn.Linear(hidden_units, n_gaussians * latent_dim)

    def forward(self, latent, action):
        self.lstm.flatten_parameters()
        sequence_len = latent.size(1)

        x = torch.cat((latent, self.embedding(action).squeeze(dim=2)), dim=2)

        h, self.hidden = self.lstm(x, self.hidden)

        pi = self.pi(h).view(-1, sequence_len, self.n_gaussians, self.latent_dim) / self.temperature
        pi = torch.softmax(pi, dim=2)

        logsigma = self.logsigma(h).view(-1, sequence_len, self.n_gaussians, self.latent_dim)
        sigma = torch.exp(logsigma)

        mu = self.mu(h).view(-1, sequence_len, self.n_gaussians, self.latent_dim)

        return mu, sigma, pi

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device

        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device)
        )


def build_rnn_model(rnn_params, latent_dim, action_size, model_path=None):
    """Builds MDN-RNN memory module, which model time dependencies.

    Args:
        rnn_params (dict): MDN-RNN parameters from .json config.
        latent_dim (int): Latent space dimensionality.
        action_size (int): Size of action shape.
        model_path (str): Path to VAE ckpt. Taken from .json config if `None` (Default: None)

    Returns:
        TorchTrainer: Compiled MDN-RNN model wrapped in TorchTrainer, ready for training.
    """

    use_cuda = torch.cuda.is_available()

    def mdn_loss_function(pred, target):
        """Mixed Density Network loss function, see:
        https://mikedusenberry.com/mixture-density-networks"""

        mu, sigma, pi = pred

        sequence_len = mu.size(1)
        latent_dim = mu.size(3)
        target = target.view(-1, sequence_len, 1, latent_dim)

        loss = Normal(loc=mu, scale=sigma)
        loss = torch.exp(loss.log_prob(target))
        loss = torch.sum(loss * pi, dim=2)
        loss = -torch.log(loss + 1e-9)

        return torch.mean(loss)

    mdn = TorchTrainer(MDN(rnn_params['hidden_units'], latent_dim, action_size,
                           rnn_params['temperature'], rnn_params['n_gaussians']),
                       device_name='cuda' if use_cuda else 'cpu')

    mdn.compile(optimizer=optim.Adam(mdn.model.parameters(), lr=rnn_params['learning_rate']),
                loss=mdn_loss_function)

    model_path = get_model_path_if_exists(
        path=model_path, default_path=rnn_params['ckpt_path'], model_name="MDN-RNN")

    if model_path is not None:
        mdn.load_ckpt(model_path)
        log.info("Loaded MDN-RNN model weights from: %s", model_path)

    return mdn
