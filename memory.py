import logging as log

import numpy as np
import h5py
import humblerl as hrl
from humblerl import Callback, Vision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import Dataset

from common_utils import get_model_path_if_exists
from third_party.torchtrainer import TorchTrainer, evaluate


class MDNVision(Vision, Callback):
    """Performs state preprocessing with VAE module and concatenates it with hidden state of MDN module.

    Args:
        vae_model (keras.Model): Keras VAE encoder.
        mdn_model (torch.nn.Module): PyTorch MDN-RNN memory.
        latent_dim (int): Latent space dimensionality.

    Note:
        In order to work, this Vision system must be also passed as callback to 'hrl.loop(...)'!
    """

    def __init__(self, vae_model, mdn_model, latent_dim):
        self.vae_model = vae_model
        self.mdn_model = mdn_model
        self.latent_dim = latent_dim

    def __call__(self, state, reward=0.):
        return self.process_state(state), reward

    def process_state(self, state):
        # NOTE: [0][0] <- it gets first in the batch latent space mean (mu)
        latent = self.vae_model.predict(state[np.newaxis, :])[0][0]
        memory = self.mdn_model.hidden[0].cpu().detach().numpy()

        # NOTE: See HRL `ply`, `on_step_taken` that would update hidden state is called AFTER
        #       Vision is used to preprocess next_state. So next_state has out-dated hidden state!
        #       What saves us is the fact, that `state` in next `ply` call will have it updated so,
        #       Transitions.state has up-to-date latent and hidden state and in all the other places
        #       exactly it is used, not next state.
        return np.concatenate((latent, memory.flatten()))

    def on_episode_start(self, episode, train_mode):
        self.mdn_model.init_hidden(1)

    def on_step_taken(self, step, transition, info):
        state = torch.from_numpy(transition.state[:self.latent_dim]).view(1, 1, -1)
        action = torch.from_numpy(np.array([transition.action])).view(1, 1, -1)
        if torch.cuda.is_available():
            state = state.cuda()
            action = action.cuda()
        with torch.no_grad(), evaluate(self.mdn_model) as net:
            net(state, action)


class MDNDataset(Dataset):
    """Dataset of sequential data to train MDN-RNN.

    Args:
        dataset_path (string): Path to HDF5 dataset file.
        sequence_len (int): Desired output sequence len.
        terminal_prob (float): Probability of sampling sequence that finishes with
            terminal state. (Default: 0.5)

    Note:
        Arrays should have the same size of the first dimension and their type should be the
        same as desired Tensor type.
    """

    def __init__(self, dataset_path, sequence_len, terminal_prob=0.5, dataset_fraction=1.):
        assert 0 < terminal_prob and terminal_prob <= 1.0, "0 < terminal_prob <= 1.0"
        assert 0 < dataset_fraction and dataset_fraction <= 1.0, "0 < dataset_fraction <= 1.0"

        self.dataset = h5py.File(dataset_path, "r")
        self.sequence_len = sequence_len
        self.terminal_prob = terminal_prob
        self.dataset_fraction = dataset_fraction
        self.latent_dim = self.dataset.attrs["LATENT_DIM"]
        self.action_dim = self.dataset.attrs["ACTION_DIM"]

    def __getitem__(self, idx):
        """Get sequence at random starting position of given sequence length from episode `idx`."""

        offset = 1

        t_start, t_end = self.dataset['episodes'][idx:idx + 2]
        episode_length = t_end - t_start
        if self.sequence_len <= episode_length - offset:
            sequence_len = self.sequence_len
        else:
            sequence_len = episode_length - offset
            log.warning(
                "Episode %d is too short to form full sequence, data will be zero-padded.", idx)

        # Sample where to start sequence of length `self.sequence_len` in episode `idx`
        # '- offset' because "next states" are offset by 'offset'
        if np.random.rand() < self.terminal_prob:
            # Take sequence ending with terminal state
            start = t_start + episode_length - sequence_len - offset
        else:
            # NOTE: np.random.randint takes EXCLUSIVE upper bound of range to sample from
            start = t_start + np.random.randint(max(1, episode_length - sequence_len - offset))

        states_ = torch.from_numpy(self.dataset['states'][start:start + sequence_len + offset])
        actions_ = torch.from_numpy(self.dataset['actions'][start:start + sequence_len])

        states = torch.zeros(self.sequence_len, self.latent_dim, dtype=states_.dtype)
        next_states = torch.zeros(self.sequence_len, self.latent_dim, dtype=states_.dtype)
        actions = torch.zeros(self.sequence_len, self.action_dim, dtype=actions_.dtype)

        # Sample latent states (this is done to prevent overfitting MDN-RNN to a specific 'z'.)
        mu = states_[:, 0]
        sigma = torch.exp(states_[:, 1] / 2)
        latent = Normal(loc=mu, scale=sigma)
        z_samples = latent.sample()

        states[:sequence_len] = z_samples[:-offset]
        next_states[:sequence_len] = z_samples[offset:]
        actions[:sequence_len] = actions_

        return [states, actions], [next_states]

    def __len__(self):
        return int(self.dataset.attrs["N_GAMES"] * self.dataset_fraction)

    def close(self):
        self.dataset.close()


class MDN(nn.Module):
    def __init__(self, hidden_units, latent_dim, action_space, temperature, n_gaussians, num_layers=1):
        super(MDN, self).__init__()

        self.hidden_units = hidden_units
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.n_gaussians = n_gaussians
        self.num_layers = num_layers

        self.embedding = nn.Embedding.from_pretrained(torch.eye(action_space.num)) \
            if isinstance(action_space, hrl.environments.Discrete) else None
        self.lstm = nn.LSTM(input_size=(latent_dim + action_space.num),
                            hidden_size=hidden_units,
                            num_layers=num_layers,
                            batch_first=True)
        self.pi = nn.Linear(hidden_units, n_gaussians * latent_dim)
        self.mu = nn.Linear(hidden_units, n_gaussians * latent_dim)
        self.logsigma = nn.Linear(hidden_units, n_gaussians * latent_dim)
        # NOTE: This is here only for backward compatibility with trained checkpoint
        self.reward = nn.Linear(hidden_units, 1)

    def forward(self, latent, action, hidden=None):
        self.lstm.flatten_parameters()
        sequence_len = latent.size(1)
        if self.embedding:
            # Use one-hot representation for discrete actions
            x = torch.cat((latent, self.embedding(action).squeeze(dim=2)), dim=2)
        else:
            # Pass raw action vector for continuous actions
            x = torch.cat((latent, action.float()), dim=2)

        h, self.hidden = self.lstm(x, hidden if hidden else self.hidden)

        pi = self.pi(h).view(-1, sequence_len, self.n_gaussians, self.latent_dim) / self.temperature
        pi = torch.softmax(pi, dim=2)

        logsigma = self.logsigma(h).view(-1, sequence_len, self.n_gaussians, self.latent_dim)
        sigma = torch.exp(logsigma)

        mu = self.mu(h).view(-1, sequence_len, self.n_gaussians, self.latent_dim)

        return mu, sigma, pi

    def sample(self, latent, action, hidden=None):
        """Sample (simulate) next state from Mixture Density Network a.k.a. Gaussian Mixture Model.

        Args:
            latent (torch.Tensor): Latent vectors to start from.
                Shape of tensor: batch x sequence x latent dim.
            action (torch.Tensor): Actions to simulate.
                Shape of tensor: batch x sequence x action dim.
            hidden (tuple): Memory module (torch.nn.LSTM) hidden state.

        Return:
            numpy.ndarray: Latent vector of next state.
                Shape of array: batch x sequence x latent dim.

        Note:
            You can find next hidden state in this module `hidden` member.
        """

        # Simulate transition
        with torch.no_grad(), evaluate(self) as net:
            mu, sigma, pi = net(latent, action, hidden)

        # Transform tensors to numpy arrays and move "gaussians mixture" dim to the end
        # NOTE: Arrays will have shape (batch x sequence x latent dim. x num. gaussians)
        mu = np.transpose(mu.cpu().detach().numpy(), axes=[0, 1, 3, 2])
        sigma = np.transpose(sigma.cpu().detach().numpy(), axes=[0, 1, 3, 2])
        pi = np.transpose(pi.cpu().detach().numpy(), axes=[0, 1, 3, 2])

        # Sample parameters of Gaussian distribution(s) from mixture
        c = pi.cumsum(axis=-1)
        u = np.random.rand(*c.shape[:-1], 1)
        choices = np.expand_dims((u < c).argmax(axis=-1), axis=-1)

        # Sample latent vector from Gaussian distribution with mean and std. dev. from above
        mean = np.take_along_axis(mu, choices, axis=-1)
        stddev = np.take_along_axis(sigma, choices, axis=-1)
        samples = mean + stddev * np.random.randn(*mean.shape)

        return np.squeeze(samples, axis=-1)

    def simulate(self, latent, actions):
        """Simulate environment trajectory.

        Args:
            latent (torch.Tensor): Latent vector with state(s) to start from.
                Shape of tensor: batch x 1 (sequence dim.) x latent dim.
            actions (torch.Tensor): Tensor with actions to take in simulated trajectory.
                Shape of tensor: batch x sequence x action dim.

        Return:
            np.ndarray: Array of latent vectors of simulated trajectory.
                Shape of array: batch x sequence x latent dim.

        Note:
            You can find next hidden state in this module `hidden` member.
        """

        states = []
        for a in range(actions.shape[1]):
            # NOTE: We use np.newaxis to preserve shape of tensor.
            states.append(self.sample(latent, actions[:, a, np.newaxis, :]))
            # NOTE: This is a bit arbitrary to set it to float32 which happens to be type of torch
            #       tensors. It can blow up further in code if we'll choose to change tensors types.
            latent = torch.from_numpy(states[-1]).float().to(next(self.parameters()).device)

        # NOTE: Squeeze former sequence dim. (which is 1 because we inferred next latent state
        #       action by action) and reorder batch dim. and list sequence dim. to finally get:
        #       batch x len(states) (sequence dim.) x latent dim.
        return np.transpose(np.squeeze(np.array(states), axis=2), axes=[1, 0, 2])

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device

        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device)
        )


def build_rnn_model(rnn_params, latent_dim, action_space, model_path=None):
    """Builds MDN-RNN memory module, which model time dependencies.

    Args:
        rnn_params (dict): MDN-RNN parameters from .json config.
        latent_dim (int): Latent space dimensionality.
        action_space (hrl.environments.ActionSpace): Action space, discrete or continuous.
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
        loss = torch.exp(loss.log_prob(target))  # TODO: Is this stable?! Check that.
        loss = torch.sum(loss * pi, dim=2)
        loss = -torch.log(loss + 1e-9)

        return torch.mean(loss)

    mdn = TorchTrainer(MDN(rnn_params['hidden_units'], latent_dim, action_space,
                           rnn_params['temperature'], rnn_params['n_gaussians']),
                       device_name='cuda' if use_cuda else 'cpu')

    mdn.compile(
        optimizer=optim.Adam(mdn.model.parameters(), lr=rnn_params['learning_rate']),
        loss=mdn_loss_function
    )

    model_path = get_model_path_if_exists(
        path=model_path, default_path=rnn_params['ckpt_path'], model_name="MDN-RNN")

    if model_path is not None:
        mdn.load_ckpt(model_path)
        log.info("Loaded MDN-RNN model weights from: %s", model_path)

    return mdn
