import datetime as dt
import logging as log
import math
import os
import random

import h5py
import humblerl as hrl
from keras.utils import Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from common_utils import get_configs, create_directory
from third_party.torchtrainer import evaluate, Callback as TorchCallback


class Config(object):
    def __init__(self, config_path, is_debug, allow_render):
        """Loads custom configuration, unspecified parameters are taken from default configuration.

        Args:
            config_path (str): Path to .json file with custom configuration
            is_debug (bool): Specify to enable debugging features
            allow_render (bool): Specify to enable render/plot features
        """

        default_config, custom_config = get_configs(config_path)

        # Merging default and custom configs, for repeating keys second dict overwrites values
        self.general = {**default_config["general"], **custom_config.get("general", {})}
        self.es = {**default_config["es_training"], **custom_config.get("es_training", {})}
        self.rnn = {**default_config["rnn_training"], **custom_config.get("rnn_training", {})}
        self.vae = {**default_config["vae_training"], **custom_config.get("vae_training", {})}
        self.is_debug = is_debug
        self.allow_render = allow_render


class StoreTransitions(hrl.Callback):
    """Save transitions to HDF5 file in four datasets:
        * 'states': States.
        * 'actions': Actions.
        * 'rewards': Rewards.
        * 'episodes': Indices of each episode (episodes[i] -> start index of episode `i`
            in states, actions and rewards datasets).

        Datasets are organized in such a way, that you can locate episode `i` by accessing
        i-th position in `episodes` to get the `start` index and (i+1)-th position to get
        the `end` index and then get all of this episode's transitions by accessing
        `states[start:end]` and `actions[start:end]`.

        HDF5 file also keeps meta-informations (attributes) as such:
        * 'N_TRANSITIONS': Datasets size (number of transitions).
        * 'N_GAMES': From how many games those transitions come from.
        * 'CHUNK_SIZE': Chunk size.
        * 'STATE_SHAPE': Shape of state.
        * 'ACTION_DIM': Action's dimensionality (1 for discrete).
    """

    def __init__(self, out_path, state_shape, action_space, min_transitions=10000, min_episodes=1000,
                 chunk_size=128, state_dtype=np.uint8, reward_dtype=np.float32):
        """Initialize memory data storage.

        Args:
            out_path (str): Path to output hdf5 file.
            state_shape (tuple): Shape of state.
            action_space (hrl.environments.ActionSpace): Object representing action space,
                check HumbleRL.
            min_transitions (int): Minimum expected number of transitions in dataset. If more is
                gathered, then hdf5 dataset size is expanded.
            min_episodes (int): Minimum expected number of episodes in dataset. If more is
                gathered, then hdf5 dataset size is expanded.
            chunk_size (int): Chunk size in transitions. For efficiency reasons, data is saved
                to file in chunks to limit the disk usage (chunk is smallest unit that get fetched
                from disk). For best performance set it to training batch size. (Default: 128)
            state_dtype (numpy.dtype): Type used to store the state (Default: np.uint8).
            reward_dtype (numpy.dtype): Type used to store the rewards (Default: np.float32).
        """

        self.out_path = out_path
        self.dataset_size = min_transitions
        self.min_transitions = min_transitions
        self.episodes_size = min_episodes
        self.state_shape = state_shape
        self.action_dim = action_space.num if isinstance(
            action_space, hrl.environments.Continuous) else 1
        self.transition_count = 0
        self.game_count = 0
        self.states = []
        self.actions = []
        self.rewards = []

        if os.path.exists(out_path):
            try:
                self.out_file = h5py.File(out_path, "a")
                self.out_states = self.out_file["states"]
                self.out_actions = self.out_file["actions"]
                self.out_rewards = self.out_file["rewards"]
                self.out_episodes = self.out_file["episodes"]
                self.transition_count = self.out_file.attrs["N_TRANSITIONS"]
                self.game_count = self.out_file.attrs["N_GAMES"]
                # NOTE: Last entry in `episodes` should point to the end of dataset - if it
                #       doesn't, it means that data gathering was interrupted mid-game and data
                #       wasn't properly saved to disk. This is a workaround and should probably
                #       be handled differently.
                if self.out_episodes[self.game_count] != self.transition_count:
                    self.game_count += 1
                    self.out_episodes[self.game_count] = self.transition_count
                return
            except KeyError:
                # File exists but isn't a proper dataset - we will create it from scratch.
                self.out_file.close()

        # Make sure that path to out file exists
        dirname = os.path.dirname(out_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        # Create output hdf5 file and fill metadata
        self.out_file = h5py.File(out_path, "w")
        self.out_file.attrs["N_TRANSITIONS"] = 0
        self.out_file.attrs["N_GAMES"] = 0
        self.out_file.attrs["CHUNK_SIZE"] = chunk_size
        self.out_file.attrs["STATE_SHAPE"] = state_shape
        self.out_file.attrs["ACTION_DIM"] = self.action_dim

        # Create datasets
        self.out_states = self.out_file.create_dataset(
            name="states", dtype=state_dtype, chunks=(chunk_size, *state_shape),
            shape=(self.dataset_size, *state_shape), maxshape=(None, *state_shape),
            compression="lzf")
        self.out_actions = self.out_file.create_dataset(
            name="actions", dtype=action_space.sample().dtype, chunks=(chunk_size, self.action_dim),
            shape=(self.dataset_size, self.action_dim), maxshape=(None, self.action_dim),
            compression="lzf")
        self.out_rewards = self.out_file.create_dataset(
            name="rewards", dtype=reward_dtype, chunks=(chunk_size,),
            shape=(self.dataset_size,), maxshape=(None,),
            compression="lzf")
        self.out_episodes = self.out_file.create_dataset(
            name="episodes", dtype=np.int, chunks=(chunk_size,),
            shape=(self.episodes_size + 1,), maxshape=(None,))

        self.out_episodes[0] = 0

    def on_step_taken(self, step, transition, info):
        action = transition.action
        self.states.append(transition.state)
        self.actions.append(action if isinstance(action, np.ndarray) else [action])
        self.rewards.append(transition.reward)

        self.transition_count += 1

        if transition.is_terminal:
            self.game_count += 1
            if self.game_count == self.episodes_size:
                self.episodes_size *= 2
                self.out_episodes.resize(self.episodes_size, axis=0)
            self.out_episodes[self.game_count] = self.transition_count

        if self.transition_count % self.min_transitions == 0:
            self._save_chunk()

    def on_loop_end(self, is_aborted):
        if len(self.states) > 0:
            self._save_chunk()

        # Close file
        self.out_file.close()

    def _save_chunk(self):
        """Save `states` and `actions` to HDF5 file. Clear the buffers.
        Update transition and games count in HDF5 file."""

        # Resize datasets if needed
        if self.transition_count > self.dataset_size:
            self.out_states.resize(self.transition_count, axis=0)
            self.out_actions.resize(self.transition_count, axis=0)
            self.out_rewards.resize(self.transition_count, axis=0)
            self.dataset_size = self.transition_count

        n_transitions = len(self.states)
        start = self.transition_count - n_transitions

        assert n_transitions > 0, "Nothing to save!"

        self.out_states[start:self.transition_count] = self.states
        self.out_actions[start:self.transition_count] = self.actions
        self.out_rewards[start:self.transition_count] = self.rewards

        self.out_file.attrs["N_TRANSITIONS"] = self.transition_count
        self.out_file.attrs["N_GAMES"] = self.game_count

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()


def convert_data_with_vae(vae_encoder, path_in, path_out, latent_dim):
    """Use trained VAE encoder to preprocess states in HDF5 dataset. The rest of the
    HDF5 file is copied without change (actions, rewards, episodes). Such a preprocessed
    dataset can be used later for Memory training.

    Args:
        vae_encoder (keras.models.Model): Trained VAE encoder.
        path_in (str): Path to HDF5 file with gathered transitions.
        path_out (str): Path to output HDF5 file with preprocessed states.
        latent_dim (int): VAE's latent state dimensionality.
    """

    with h5py.File(path_in, "r") as hdf_in, h5py.File(path_out, "w") as hdf_out:
        # Copy datasets and params from input HDF5, excluding the states
        hdf_in.copy("actions", hdf_out)
        hdf_in.copy("rewards", hdf_out)
        hdf_in.copy("episodes", hdf_out)
        hdf_out.attrs["N_TRANSITIONS"] = hdf_in.attrs["N_TRANSITIONS"]
        hdf_out.attrs["N_GAMES"] = hdf_in.attrs["N_GAMES"]
        hdf_out.attrs["CHUNK_SIZE"] = hdf_in.attrs["CHUNK_SIZE"]
        hdf_out.attrs["ACTION_DIM"] = hdf_in.attrs["ACTION_DIM"]

        hdf_out.attrs["LATENT_DIM"] = latent_dim
        # 2 because latent space mean (mu) and logvar are saved
        hdf_out.attrs["STATE_SHAPE"] = [2, latent_dim]

        n_transitions = hdf_in.attrs["N_TRANSITIONS"]
        chunk_size = hdf_in.attrs["CHUNK_SIZE"]
        new_states = hdf_out.create_dataset(
            name="states", dtype=np.float32, chunks=(chunk_size, 2, latent_dim),
            shape=(n_transitions, 2, latent_dim), maxshape=(None, 2, latent_dim),
            compression="lzf")

        # Preprocess states from input dataset by using VAE
        log.info("Preprocessing states with VAE...")
        n_chunks = math.ceil(n_transitions / chunk_size)
        pbar = tqdm(range(n_chunks), ascii=True)
        for i in pbar:
            beg, end = i * chunk_size, min((i + 1) * chunk_size, n_transitions)
            # Grab a batch of states and feed it to VAE
            # NOTE: [:2] <- gets latent space mean (mu) and logvar, then swaps axes from
            #       [2, batch_size, latent_dim] into [batch_size, 2, latent_dim].
            states_batch = hdf_in["states"][beg:end]
            new_states[beg:end] = np.swapaxes(vae_encoder.predict(states_batch / 255.)[:2], 0, 1)


class HDF5DataGenerator(Sequence):
    """Generates data for Keras model from bug HDF5 files."""

    def __init__(self, hdf5_path, dataset_X, dataset_y, batch_size,
                 start=None, end=None, preprocess_fn=None):
        """Initialize data generator.

        Args:
            hdf5_path (str): Path to HDF5 file with data.
            dataset_X (str): Dataset's name with data.
            dataset_y (str): Dataset's name with targets.
            batch_size (int): Size of batch to return.
            start (int): Index where to start (inclusive) reading data/targets from dataset.
                If `None`, then it starts from the beginning. (Default: None)
            end (int): Index where to end (exclusive) reading data/targets from dataset.
                If `None`, then it reads dataset to the end. (Default: None)
            preprocess_fn (func): Function which accepts two arguments (batch of data and targets).
                It should return preprocessed batch (two values, data and targets!). If `None`, then
                no preprocessing is done. (Default: None)
        """

        hfile = h5py.File(hdf5_path, 'r')
        self.X = hfile[dataset_X]
        self.y = hfile[dataset_y]
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn

        if start is None:
            self.start = 0
        else:
            self.start = start

        if end is None:
            self.end = len(self.X)
        else:
            self.end = end

    def __len__(self):
        """Denotes the number of batches per epoch.

        Return:
            int: Number of batches in epoch.
        """

        return int(np.ceil((self.end - self.start) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data.

        Args:
            idx (int): Batch index.

        Return:
            np.ndarray: Batch of training examples.
            np.ndarray: Batch of targets.
        """

        start = self.start + idx * self.batch_size
        end = min(start + self.batch_size, self.end)

        X = self.X[start:end]
        y = self.y[start:end]

        if self.preprocess_fn is not None:
            X, y = self.preprocess_fn(X, y)

        return X, y


def create_generating_agent(generating_agent, env):
    """Create an agent that will generate data for VAE/MEM training.
    By default, a random agent is created. Some agents may require additional
    callbacks to be added to hrl.loop.

    Args:
        generating_agent (str): Generating agent to create.
        env (hrl.Environment):  Game's environment.

    Returns:
        hrl.Mind: Generating agent.
        list:     Callbacks that should be added to hrl.loop, empty list by default.
    """

    if generating_agent == 'car_racing':
        log.info("Created agent for Car Racing.")
        agent = CarRacingAgent(env)
        return agent, [agent.step_counter_callback]
    elif generating_agent == 'move_n_push':
        log.info("Created agent for Sokoban.")
        agent = MoveNPushAgent(env)
        return agent, []
    log.info("Created generic random agent.")
    return hrl.agents.RandomAgent(env), []


class EnvironmentStepCounter(hrl.Callback):
    """Callback for keeping track of current step in the environment."""

    def __init__(self):
        self.step_counter = 0

    def on_episode_start(self, episode, train_mode):
        self.step_counter = 0

    def on_step_taken(self, step, transition, info):
        self.step_counter += 1


class CarRacingAgent(hrl.Mind):
    """'Random' agent for CarRacing game. Normal random agent doesn't work well,
    since actions need to be repeated for some steps, for the car to move somewhat
    sensibly + it needs to accelerate first.

    Adapted from: https://github.com/AppliedDataSciencePartners/WorldModels
    """

    def __init__(self, env):
        self.env = env
        self.current_action = env.action_space.sample()
        self.step_counter_callback = EnvironmentStepCounter()

    def plan(self, state, train_mode, debug_mode):
        action = self.current_action
        current_step = self.step_counter_callback.step_counter

        # Accelerate for first 60 steps to get the car moving
        if current_step < 60:
            action = np.array([0, 1, 0])

        # Change action every 5 steps
        if current_step % 5 == 0:
            rn = random.randint(0, 9)
            if rn in [0]:
                action = np.array([0, 0, 0])
            if rn in [1, 2, 3, 4]:
                action = np.array([0, random.random(), 0])
            if rn in [5, 6, 7]:
                action = np.array([-random.random(), 0, 0])
            if rn in [8]:
                action = np.array([random.random(), 0, 0])
            if rn in [9]:
                action = np.array([0, 0, random.random()])

        self.current_action = action
        return action


class MoveNPushAgent(hrl.Mind):
    """'Random' agent for Sokoban game. It performs push action (in random direction) with
    probability 0.7 and move action otherwise.
    """

    def __init__(self, env):
        self.action_num = env.action_space.num

    def plan(self, state, train_mode, debug_mode):
        action = np.random.randint(4) + 4 * (np.random.rand() >= 0.7)
        one_hot = np.zeros(self.action_num)
        one_hot[action] = 1
        return one_hot


class MemoryVisualization(TorchCallback):
    """Render simulated experience of memory module.

    Args:
        config (Config): Configuration loaded json .from file.
        vae_decoder (keras.models.Model): Vision decoder Keras model.
        mem_model (torch.nn.Module): PyTorch memory module.
        dataset (torch.utils.data.Dataset): PyTroch dataset with data from ExperienceStorage.
        dir_name (string): Directory name where plots will be saved. (Default: 'plots')
    """

    def __init__(self, config, vae_decoder, mem_model, dataset, dir_name='plots'):
        self.config = config
        self.decoder = vae_decoder
        self.model = mem_model
        self.sequence_len = self.config.rnn['sequence_len']
        self.latent_dim = self.config.vae['latent_space_dim']

        # Check if destination dir exists
        self.plots_dir = os.path.join(self.config.rnn['logs_dir'], dir_name)
        create_directory(self.plots_dir)

        # Prepare data
        (states, actions), _ = dataset[0]
        self.n_episodes = min(self.config.rnn['rend_n_episodes'], len(dataset))
        self.eval_states = torch.zeros((self.n_episodes, self.sequence_len, states.shape[1]),
                                       device=next(self.model.parameters()).device,
                                       dtype=states.dtype)
        self.eval_actions = torch.zeros((self.n_episodes, self.sequence_len, actions.shape[1]),
                                        device=next(self.model.parameters()).device,
                                        dtype=actions.dtype)
        for i in range(self.n_episodes):
            (states, actions), _ = dataset[i]
            self.eval_states[i] = states
            self.eval_actions[i] = actions

    def on_epoch_begin(self, _):
        with evaluate(self.model) as net:
            # Initialize memory module
            net.init_hidden(self.n_episodes)

            # Initialize hidden state (warm-up memory module)
            seq_half = self.sequence_len // 2
            with torch.no_grad():
                net(self.eval_states[:, :seq_half], self.eval_actions[:, :seq_half])

        orig_mu = self.eval_states[:, seq_half, :]
        pred_mu = self.model.simulate(
            torch.unsqueeze(orig_mu, 1),  # Add sequence dim.
            self.eval_actions[:, seq_half:seq_half + \
                              self.config.rnn["rend_n_rollouts"] * self.config.rnn["rend_step"], :]
        ).reshape(-1, self.latent_dim)

        orig_img = self.decoder.predict(orig_mu.cpu().detach().numpy())[:, np.newaxis]
        pred_img = self.decoder.predict(pred_mu[::self.config.rnn["rend_step"]]).reshape(
            self.n_episodes, self.config.rnn["rend_n_rollouts"], *self.config.general['state_shape'])

        samples = np.concatenate((orig_img, pred_img), axis=1)

        fig = plt.figure(figsize=(
            self.config.rnn["rend_n_rollouts"] + 1,
            self.n_episodes + 1))  # Add + 1 to make space for titles
        gs = gridspec.GridSpec(self.n_episodes,
                               self.config.rnn["rend_n_rollouts"] + 1,
                               wspace=0.05, hspace=0.05, figure=fig)

        for i in range(self.n_episodes):
            for j in range(self.config.rnn["rend_n_rollouts"] + 1):
                ax = plt.subplot(gs[i, j])
                plt.axis('off')
                ax.set_aspect('equal')
                if i == 0:
                    if j == 0:
                        ax.set_title("start")
                    else:
                        ax.set_title("t + {}".format(j * self.config.rnn["rend_step"]))
                plt.imshow(samples[i, j])

        # Save figure to logs dir
        plt.savefig(os.path.join(
            self.plots_dir,
            "memory_sample_{}".format(dt.datetime.now().strftime("%d-%mT%H:%M:%S"))
        ))
        plt.close()
