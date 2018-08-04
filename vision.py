import keras.backend as K
import logging as log

from keras.layers import Activation, Conv2D, Conv2DTranspose, Dense, Flatten, Input, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam


IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 64, 64, 3


def build_vae_model(vae_params):
    """Builds VAE encoder, decoder using Keras Model and VAE loss.

    Args:
        vae_params (dict): VAE parameters from .json config.

    Returns:
        keras.models.Model: Compiled VAE, ready for training.
        keras.models.Model: Encoder.
        keras.models.Model: Decoder.
    """

    if K.image_data_format() == 'channel_first':
        raise ValueError("Channel first backends aren't supported!")

    ### Encoder img -> mu, logvar ###

    encoder_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    h = Conv2D(32, activation='relu', kernel_size=4, strides=2)(encoder_input)  # -> 31x31x32
    h = Conv2D(64, activation='relu', kernel_size=4, strides=2)(h)              # -> 14x14x64
    h = Conv2D(128, activation='relu', kernel_size=4, strides=2)(h)             # -> 6x6x128
    h = Conv2D(256, activation='relu', kernel_size=4, strides=2)(h)             # -> 2x2x256

    batch_size = K.shape(h)[0]  # Needed to sample latent vector
    h_shape = K.int_shape(h)    # Needed to reconstruct in decoder

    h = Flatten()(h)
    mu = Dense(vae_params['latent_space_dim'])(h)
    logvar = Dense(vae_params['latent_space_dim'])(h)

    ### Sample latent vector ###

    def sample(args):
        mu, logvar = args
        return mu + K.exp(logvar) * K.random_normal(
            shape=(batch_size, vae_params['latent_space_dim']))

    z = Lambda(sample, output_shape=(vae_params['latent_space_dim'],))([mu, logvar])

    encoder = Model(encoder_input, [mu, z], name='Encoder')
    encoder.summary(print_fn=lambda x: log.debug('%s', x))

    ### Decoder z -> img ###

    decoder_input = Input(shape=(vae_params['latent_space_dim'],))

    h = Reshape(h_shape[1:])(
        Dense(h_shape[1] * h_shape[2] * h_shape[3], activation='relu')(decoder_input)
    )

    h = Conv2DTranspose(128, activation='relu', kernel_size=4, strides=2)(h)     # -> 6x6x128
    h = Conv2DTranspose(64, activation='relu', kernel_size=4, strides=2)(h)      # -> 14x14x64
    h = Conv2DTranspose(32, activation='relu', kernel_size=4, strides=2)(h)      # -> 30x30x32
    out = Conv2DTranspose(3, activation='sigmoid', kernel_size=6, strides=2)(h)  # -> 64x64x3

    decoder = Model(decoder_input, out, name='Decoder')
    decoder.summary(print_fn=lambda x: log.debug('%s', x))

    ### VAE loss ###

    def elbo_loss(target, pred):
        # NOTE: You use K.reshape to preserve batch dim. K.flatten doesn't work like flatten layer
        #       and flatten batch dim. too!
        # NOTE 2: K.binary_crossentropy does element-wise crossentropy as you need (it calls
        #         tf.nn.sigmoid_cross_entropy_with_logits in backend), but Keras loss
        #         binary_crossentropy would average over spatial dim. You sum it as you don't want
        #         to weight reconstruction loss lower (divide by H * W * C) then KL loss.
        reconstruction_loss = K.sum(
            K.binary_crossentropy(
                K.reshape(target, [batch_size, -1]), K.reshape(pred, [batch_size, -1])
            ),
            axis=1
        )

        # NOTE: Closed form of KL divergence for Gaussians.
        #       See Appendix B from VAE paper (Kingma 2014):
        #       https://arxiv.org/abs/1312.6114
        KL_loss = K.sum(
            1. + logvar - K.square(mu) - K.exp(logvar),
            axis=1
        ) / 2

        return reconstruction_loss - KL_loss

    ### Build and compile VAE model ###

    decoder_output = decoder(encoder(encoder_input)[1])
    vae = Model(encoder_input, decoder_output, name='VAE')
    vae.compile(optimizer=Adam(lr=vae_params['learning_rate']), loss=elbo_loss)
    vae.summary(print_fn=lambda x: log.debug('%s', x))

    return vae, encoder, decoder
