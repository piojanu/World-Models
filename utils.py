import json
import h5py as h5
import numpy as np
import os
import logging as log

from keras.utils import Sequence
from skimage.transform import resize
from tqdm import tqdm


class Config(object):
    def __init__(self, config_path, is_debug, allow_render):
        """Loads custom configuration, unspecified parameters are taken from default configuration.

        Args:
            config_path (str): Path to .json file with custom configuration
            is_debug (bool): Specify to enable debugging features
            allow_render (bool): Specify to enable render/plot features
        """

        with open("config.json.dist") as f:
            default_config = json.loads(f.read())
        with open(config_path) as f:
            custom_config = json.loads(f.read())

        # Merging default and custom configs, for repeating keys second dict overwrites values
        self.general = {**default_config["general"], **custom_config.get("general", {})}
        self.es = {**default_config["es_training"], **custom_config.get("es_training", {})}
        self.rnn = {**default_config["rnn_training"], **custom_config.get("rnn_training", {})}
        self.vae = {**default_config["vae_training"], **custom_config.get("vae_training", {})}
        self.is_debug = is_debug
        self.allow_render = allow_render


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

        hfile = h5.File(hdf5_path, 'r')
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


class TqdmStream(object):
    @classmethod
    def write(_, msg):
        tqdm.write(msg, end='')

    @classmethod
    def flush(_):
        pass


def state_processor(img, state_shape, crop_range):
    """Resize states to `state_shape` with cropping of `crop_range`.

    Args:
        img (np.ndarray): Image to crop and resize.
        state_shape (tuple): Output shape. Default: [64, 64, 3]

        crop_range (string): Range to crop as indices of array. Default: "[30:183, 28:131, :]"
    Return:
        np.ndarray: Cropped and reshaped to `state_shape` image.
    """

    # Crop image to `crop_range`, removes e.g. score bar
    img = eval("img" + crop_range)

    # Resize to 64x64 and cast to 0..255 values if requested
    return resize(img, state_shape, mode='constant') * 255


def get_model_path_if_exists(path, default_path, model_name):
    """Resize states to `state_shape` with cropping of `crop_range`.

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
            log.info("{} weights in \"{}\" doesn't exist! Starting tabula rasa.".format(model_name, path))
    elif not os.path.exists(path):
        raise ValueError("{} weights in \"{}\" path doesn't exist!".format(model_name, path))
    return path
