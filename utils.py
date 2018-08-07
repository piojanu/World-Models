import json
import logging as log

import h5py as h5
import numpy as np

from keras.utils import Sequence
from skimage.transform import resize


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


def pong_state_processor(img):
    """Resize states to 64x64 with cropping suited for Pong.

    Args:
        img (np.ndarray): Image to crop and resize.

    Return:
        np.ndarray: Cropped and reshaped to 64x64px image.
    """

    # Crop image to 160x160x3, removes e.g. score bar
    img = img[35:195, :, :]

    # Resize to 64x64 and cast to 0..255 values
    return resize(img, (64, 64)) * 255


def boxing_state_processor(img):
    """Resize states to 64x64 with cropping suited for Boxing.

    Args:
        img (np.ndarray): Image to crop and resize.

    Return:
        np.ndarray: Cropped and reshaped to 64x64px image.
    """

    # Crop image to 153x103x3, removes e.g. score bar
    img = img[30:183, 28:131, :]

    # Resize to 64x64
    return resize(img, (64, 64)) * 255
