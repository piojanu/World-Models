from collections import deque
import json
import logging as log
import os
from pickle import Pickler, Unpickler

from humblerl import Callback
from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class Storage(Callback):
    """Storage train examples.

    Args:
        out_path (str): Path to output hdf5 file.
        exp_replay_size (int): How many transitions to keep at max. If this number is exceeded,
            oldest transition is dropped.
        gamma (float): Discount factor.
    """

    def __init__(self, out_path, exp_replay_size, gamma):
        self.small_bag = deque()
        self.big_bag = deque()

        self.out_path = out_path
        self.exp_replay_size = exp_replay_size
        self.gamma = gamma

        self._recent_action_probs = None

    def on_action_planned(self, step, logits, info):
        # Proportional without temperature
        self._recent_action_probs = logits / np.sum(logits)

    def on_step_taken(self, step, transition, info):
        # NOTE: We never pass terminal state (it would be next_state), so NN can't learn directly
        #       what is the value of terminal/end state.
        self.small_bag.append(self._create_small_package(transition))
        if len(self.small_bag) > self.exp_replay_size:
            self.small_bag.popleft()

        if transition.is_terminal:
            return_t = 0
            for state, reward, mcts_pi in reversed(self.small_bag):

                return_t = reward + self.gamma * return_t
                self.big_bag.append((state, mcts_pi, return_t))

                if len(self.big_bag) > self.exp_replay_size:
                    self.big_bag.popleft()

            self.small_bag.clear()

    def store(self):
        path = self.out_path
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            log.warning("Examples store directory does not exist! Creating directory %s", folder)
            os.makedirs(folder)

        with open(path, "wb+") as f:
            Pickler(f).dump(self.big_bag)

    def load(self):
        path = self.out_path
        if not os.path.isfile(path):
            log.warning("File with train examples was not found.")
        else:
            log.info("File with train examples found. Reading it.")
            with open(path, "rb") as f:
                self.big_bag = Unpickler(f).load()

            # Prune dataset if too big
            while len(self.big_bag) > self.exp_replay_size:
                self.big_bag.popleft()

    @property
    def metrics(self):
        logs = {"# samples": len(self.big_bag)}
        return logs

    def _create_small_package(self, transition):
        return (transition.state, transition.reward, self._recent_action_probs)


class ReturnTracker(Callback):
    """Tracks return."""

    def on_episode_start(self, episode, train_mode):
        self.ret = 0

    def on_step_taken(self, step, transition, info):
        self.ret += transition.reward

    @property
    def metrics(self):
        return {"return": self.ret}


class TensorBoardLogger(object):
    """Logging in TensorBoard without TensorFlow ops.

    https://gist.github.com/1f8dfb1b5c82627ae3efcfbbadb9f514.git
    Simple example on how to log scalars and images to tensorboard without tensor ops.

    License: Copyleft
    Author: Michael Gygli
    """

    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.

        Args:
            tag (basestring): Name of the scalar.
            value (number): Value to log.
            step (int): Training iteration.
        """

        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


class TqdmStream(object):
    @classmethod
    def write(_, msg):
        tqdm.write(msg, end='')

    @classmethod
    def flush(_):
        pass


def get_configs(config_path):
    """Loads default and custom configs
        Args:
            config_path (str): Path to .json file with custom configuration

        Return:
            dict: Default config
            dict: Custom config
    """

    with open(os.path.join(os.path.dirname(__file__), "config.json.dist")) as config_file:
        default_config = json.loads(config_file.read())

    if os.path.exists(config_path):
        with open(config_path) as custom_config_file:
            custom_config = json.loads(custom_config_file.read())
    else:
        custom_config = {}
    return default_config, custom_config


def obtain_config(ctx, use_gpu=True):
    if use_gpu:
        limit_gpu_memory_usage()
    else:
        force_cpu()

    return ctx.obj


def limit_gpu_memory_usage():
    """This function makes that we don't allocate more graphics memory than we need.
       For TensorFlow, we need to set `alow_growth` flag to True.
       For PyTorch, this is the default behavior.

    """

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))


def create_directory(dirname):
    """Create directory recursively, if it doesn't exit

    Args:
        dirname (str): Name of directory (path, e.g. "path/to/dir/")
    """

    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)


def force_cpu():
    """Force using CPU"""

    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def mute_tf_logs_if_needed():
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_model_path_if_exists(path, default_path, model_name):
    """Check if path (default_path) exist and choose one.

    Args:
        path (string): Specified path to model
        default_path (string): Specified path to model
        model_name (string): Model name ie. VAE

    Returns:
        Path to model or None, depends whether first or second path exist
    """

    if path is None:
        if os.path.exists(default_path):
            path = default_path
        else:
            log.info("%s weights in \"%s\" doesn't exist! Starting tabula rasa.",
                     model_name, default_path)
    elif not os.path.exists(path):
        raise ValueError("{} weights in \"{}\" path doesn't exist!".format(model_name, path))

    return path
