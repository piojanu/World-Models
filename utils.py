import json

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
