from skimage.transform import resize


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
