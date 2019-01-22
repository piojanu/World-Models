import cma
import logging as log
import numpy as np
import os.path
import pickle
import humblerl as hrl

from humblerl import ChainVision, Mind, Worker
from memory import build_rnn_model, MDNVision
from vision import BasicVision, build_vae_model

from common_utils import ReturnTracker, create_directory, get_model_path_if_exists


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class CMAES:
    """Agent using CMA-ES algorithm.

    Args:
        n_params (int)       : Number of model parameters (NN weights).
        sigma_init (float)   : Initial standard deviation. (Default: 0.1)
        popsize (int)        : Population size. (Default: 100)
        weight_decay (float) : L2 weight decay rate. (Default: 0.01)
    """

    def __init__(self, n_params, sigma_init=0.1, popsize=100, weight_decay=0.01):
        self.weight_decay = weight_decay
        self.population = None
        self.es = cma.CMAEvolutionStrategy(n_params * [0], sigma_init, {'popsize': popsize})
        self.best_score = -np.inf

    def check_if_better(self, new_best_score):
        """If new score is better than current then update self best score"""

        do_update = new_best_score > self.best_score
        if do_update:
            self.best_score = new_best_score
        return do_update

    def ask(self):
        """Returns a list of parameters for new population."""
        self.population = np.array(self.es.ask())
        return self.population

    def tell(self, returns):
        reward_table = np.array(returns)
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.population)
            reward_table -= l2_decay
        # Convert minimizer to maximizer.
        self.es.tell(self.population, (-1 * reward_table).tolist())

    def current_param(self):
        return self.es.result[5]  # mean solution, presumably better with noise

    def best_param(self):
        return self.es.result[0]  # best evaluated solution

    def save_es_ckpt_and_mind_weights(self, ckpt_path, mind_path, score):
        # Create CMA-ES checkpoint dir if doesn't exist
        create_directory(os.path.dirname(ckpt_path))

        # Create Mind weights checkpoint dir if doesn't exist
        mind_dir = os.path.dirname(mind_path)
        create_directory(mind_dir)

        # Create paths for best and mean Mind weights checkpoints
        mind_name = os.path.basename(mind_path).split('.')[0]
        best_path = os.path.join(mind_dir, mind_name + "_best.ckpt")
        mean_path = os.path.join(mind_dir, mind_name + "_mean.ckpt")

        with open(os.path.abspath(ckpt_path), 'wb') as f:
            pickle.dump(self, f)
        log.debug("Saved CMA-ES checkpoint in path: %s", ckpt_path)

        if self.check_if_better(score):
            log.info("New best score: %f", score)
            with open(os.path.abspath(best_path), 'wb') as f:
                pickle.dump(self.best_param(), f)
            with open(os.path.abspath(mean_path), 'wb') as f:
                pickle.dump(self.current_param(), f)
            log.debug("Saved Mind weights in path: %s", mind_path)

    @staticmethod
    def load_ckpt(path):
        with open(os.path.abspath(path), 'rb') as f:
            return pickle.load(f)


class Evaluator(Worker):
    def __init__(self, config, state_size, action_space, vae_path, mdn_path):
        self.config = config
        self.state_size = state_size
        self.action_space = action_space
        self.vae_path = vae_path
        self.mdn_path = mdn_path

        self._env = None
        self._basic_vision = None
        self._mdn_vision = None

    def initialize(self):
        self._env = hrl.create_gym(self.config.general['game_name'])
        self._basic_vision, self._mdn_vision = self._vision_factory()

    def mind_factory(self, weights):
        mind = LinearModel(self.state_size, self.action_space)
        mind.set_weights(weights)
        return mind

    @property
    def callbacks(self):
        return [ReturnTracker(), self._mdn_vision]

    @property
    def vision(self):
        return ChainVision(self._basic_vision, self._mdn_vision)

    def _vision_factory(self):
        # Build VAE model and load checkpoint
        _, encoder, _ = build_vae_model(self.config.vae,
                                        self.config.general['state_shape'],
                                        self.vae_path)

        # Build MDN-RNN model and load checkpoint
        rnn = build_rnn_model(self.config.rnn,
                              self.config.vae['latent_space_dim'],
                              self.action_space,
                              self.mdn_path)

        return (BasicVision(
            state_shape=self.config.general['state_shape'],
            crop_range=self.config.general['crop_range']),
            MDNVision(encoder, rnn.model, self.config.vae['latent_space_dim']))


class LinearModel(Mind):
    """Simple linear regression agent."""

    def __init__(self, input_dim, action_space):
        self.in_dim = input_dim
        self.action_space = action_space
        self.out_dim = action_space.num
        self.is_discrete = isinstance(action_space, hrl.environments.Discrete)

        self.weights = np.zeros((self.in_dim + 1, self.out_dim))

    def plan(self, state, train_mode, debug_mode):
        action_vec = np.concatenate((state, [1.])) @ self.weights
        # Discrete: Treat action_vec as logits and pass them to humblerl
        # Continuous: Treat action_vec as action to perform and use tanh
        #             to bound its values to [-1, 1]
        return action_vec if self.is_discrete else np.tanh(action_vec)

    def set_weights(self, weights):
        self.weights[:] = weights.reshape(self.in_dim + 1, self.out_dim)

    @property
    def n_weights(self):
        return (self.in_dim + 1) * self.out_dim

    @staticmethod
    def load_weights(path):
        with open(os.path.abspath(path), 'rb') as f:
            return pickle.load(f)


def build_mind(es_params, input_dim, action_space, model_path):
    """Builds linear regression controller model.

    Args:
        es_params (dict): CMA-ES training parameters from .json config.
        input_dim (int): Should be vision latent space dim. + memory hidden state size.
        action_space (hrl.environments.ActionSpace): Action space, discrete or continuous.
        model_path (str): Path to Mind weights.

    Returns:
        LinearModel: HumbleRL 'Mind' with weights loaded from file if available.
    """

    mind = LinearModel(input_dim, action_space)
    mind.set_weights(LinearModel.load_weights(path=model_path))
    log.info("Loaded Mind weights from: %s", model_path)

    return mind


def build_es_model(es_params, n_params, model_path=None):
    """Builds CMA-ES solver.

    Args:
        es_params (dict): CMA-ES training parameters from .json config.
        n_params (int): Number of parameters for CMA-ES.
        model_path (str): Path to CMA-ES ckpt. Taken from .json config if `None` (Default: None)

    Returns:
        CMAES: CMA-ES solver ready for training.
    """

    model_path = get_model_path_if_exists(
        path=model_path, default_path=es_params['ckpt_path'], model_name="CMA-ES")

    if model_path is not None:
        solver = CMAES.load_ckpt(model_path)
        log.info("Loaded CMA-ES parameters from: %s", model_path)
    else:
        solver = CMAES(
            n_params=n_params, popsize=es_params['popsize'], weight_decay=es_params['l2_decay'])
        log.info("CMA-ES parameters in \"%s\" doesn't exist! "
                 "Created solver with pop. size: %d and l2 decay: %f.",
                 es_params['ckpt_path'], es_params['popsize'], es_params['l2_decay'])

    return solver
