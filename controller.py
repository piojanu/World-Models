import cma
import logging as log
import numpy as np
import os.path
import pickle
import bz2
import third_party.humblerl as hrl

from functools import partial
from third_party.humblerl import Callback, Mind, Worker
from memory import build_rnn_model, MDNVision
from utils import state_processor
from vision import build_vae_model

from utils import get_model_path_if_exists


def compute_ranks(x):
    """Computes fitness ranks in rage: [0, len(x))."""
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """Computes ranks and normalize them by the number of samples.
       Finally scale them to the range [−0.5,0.5]"""
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class CMAES:
    """Agent using CMA-ES algorithm."""

    def __init__(self, n_params, sigma_init=0.1, popsize=100, weight_decay=0.01):
        """Initialize CMA-ES agent.

        Args:
            n_params (int)       : Number of model parameters (NN weights).
            sigma_init (float)   : Initial standard deviation. (Default: 0.1)
            popsize (int)        : Population size. (Default: 100)
            weight_decay (float) : L2 weight decay rate. (Default: 0.01)
        """

        self.weight_decay = weight_decay
        self.population = None

        self.es = cma.CMAEvolutionStrategy(n_params * [0], sigma_init, {'popsize': popsize})

    def ask(self):
        """Returns a list of parameters for new population."""
        self.population = np.array(self.es.ask())
        return self.population

    def tell(self, returns):
        reward_table = np.array(returns)
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.population)
            reward_table -= l2_decay
        # Apply fitness shaping function
        reward_table = compute_centered_ranks(reward_table)
        # Convert minimizer to maximizer.
        self.es.tell(self.population, (-1 * reward_table).tolist())

    def current_param(self):
        return self.es.result[5]  # mean solution, presumably better with noise

    def best_param(self):
        return self.es.result[0]  # best evaluated solution

    def save_ckpt(self, path):
        with bz2.BZ2File(os.path.abspath(path), 'w') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_ckpt(path):
        with bz2.BZ2File(os.path.abspath(path), 'r') as f:
            return pickle.load(f)


class Evaluator(Worker):
    def __init__(self, config, state_size, action_size, vae_path, mdn_path):
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.vae_path = vae_path
        self.mdn_path = mdn_path

        self._env = None
        self._vision = None

    def initialize(self):
        self._env = hrl.create_gym(self.config.general['game_name'])
        self._vision = self._vision_factory()

    def mind_factory(self, weights):
        mind = LinearModel(self.state_size, self.action_size)
        mind.set_weights(weights)
        return mind

    @property
    def callbacks(self):
        return [ReturnTracker(), self.vision]

    @property
    def vision(self):
        return self._vision

    def _vision_factory(self):
        # Build VAE model and load checkpoint
        _, encoder, _ = build_vae_model(self.config.vae,
                                        self.config.general['state_shape'],
                                        self.vae_path)

        # Build MDN-RNN model and load checkpoint
        rnn = build_rnn_model(self.config.rnn,
                              self.config.vae['latent_space_dim'],
                              self.action_size,
                              self.mdn_path)

        # Resizes states to `state_shape` with cropping and encode to latent space + hidden state
        return MDNVision(encoder, rnn.model, self.config.vae['latent_space_dim'],
                         state_processor_fn=partial(
                             state_processor,
                             state_shape=self.config.general['state_shape'],
                             crop_range=self.config.general['crop_range']))


class LinearModel(Mind):
    """Simple linear regression agent."""

    def __init__(self, input_dim, output_dim):
        self.in_dim = input_dim
        self.out_dim = output_dim

        self.weights = np.zeros((self.in_dim + 1, self.out_dim))

    def plan(self, state, player, train_mode, debug_mode):
        return np.concatenate((state, [1.])) @ self.weights

    def set_weights(self, weights):
        self.weights[:] = weights.reshape(self.in_dim + 1, self.out_dim)

    @property
    def n_weights(self):
        return (self.in_dim + 1) * self.out_dim


class ReturnTracker(Callback):
    """Tracks return."""

    def on_episode_start(self, episode, train_mode):
        self.ret = 0

    def on_step_taken(self, step, transition, info):
        self.ret += transition.reward

    @property
    def metrics(self):
        return {"return": self.ret}


def build_es_model(es_params, input_dim, action_size, model_path=None):
    """Builds linear regression controller model and CMA-ES solver.

    Args:
        es_params (dict): CMA-ES training parameters from .json config.
        input_dim (int): Should be vision latent space dim. + memory hidden state size.
        action_size (int): Size of action shape.
        model_path (str): Path to CMA-ES ckpt. Taken from .json config if `None` (Default: None)

    Returns:
        CMAES: CMA-ES solver ready for training.
        LinearModel: HumbleRL 'Mind' with weights set to ones from checkpoint if available.
    """

    mind = LinearModel(input_dim, action_size)

    model_path = get_model_path_if_exists(
        path=model_path, default_path=es_params['ckpt_path'], model_name="CMA-ES")

    if model_path is not None:
        solver = CMAES.load_ckpt(model_path)
        log.info("Loaded CMA-ES parameters from: %s", model_path)
    else:
        solver = CMAES(
            mind.n_weights, popsize=es_params['popsize'], weight_decay=es_params['l2_decay'])
        log.info("CMA-ES parameters in \"%s\" doesn't exist! ",
                 "Created solver with pop. size: %d and l2 decay: %f.",
                 es_params['ckpt_path'], es_params['popsize'], es_params['l2_decay'])

    mind.set_weights(solver.current_param())
    return solver, mind
