#!/usr/bin/env python3
import click
import datetime as dt
import logging as log
import numpy as np
import os
import third_party.humblerl as hrl

from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from keras.utils import HDF5Matrix
from third_party.humblerl.utils import RandomAgent
from third_party.humblerl.callbacks import StoreTransitions2Hdf5
from utils import Config, boxing_state_processor as state_processor
from vision import build_vae_model

STATE_SHAPE = (64, 64, 3)


@click.group()
@click.pass_context
@click.option('-c', '--config_path', type=click.Path(exists=True), default="config.json",
              help="Path to configuration file (Default: config.json)")
@click.option('--debug/--no-debug', default=False, help="Enable debug logging (Default: False)")
@click.option('--render/--no-render', default=False, help="Allow to render/plot (Default: False)")
def cli(ctx, config_path, debug, render):
    # Get and set up logger level and formatter
    log.basicConfig(level=log.DEBUG if debug else log.INFO, format="[%(levelname)s]: %(message)s")

    # Load configuration from .json file into ctx object
    ctx.obj = Config(config_path, debug, render)


@cli.command()
@click.pass_context
@click.argument('path', type=click.Path(exists=True), required=True)
def train_vae(ctx, path):
    """Train VAE model as specified in .json config with data at `PATH`."""

    config = ctx.obj

    # Get training data
    X_train = HDF5Matrix(path, 'states', normalizer=lambda x: x / 255.)

    # Build VAE model
    vae, _, _ = build_vae_model(config.vae)

    # If render features enabled...
    if config.allow_render:
        # ...plot first eight training examples with VAE reconstructions
        # at the beginning of every epoch
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        # Check if destination dir exists
        plots_dir = os.path.join(config.vae['logs_dir'], "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Evaluate VAE at the end of epoch
        def plot_samples(epoch, logs):
            X_eval = X_train[:8]
            pred = vae.predict(X_eval)

            samples = np.empty_like(np.concatenate((X_eval, pred)))
            samples[0::2] = X_eval
            samples[1::2] = pred

            _ = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(*STATE_SHAPE))

            # Save figure to logs dir
            plt.savefig(os.path.join(
                plots_dir,
                "vision_sample_{}".format(dt.datetime.now().strftime("%d-%mT%H:%M"))
            ))
    else:
        def plot_samples(epoch, logs):
            pass

    # Initialize callbacks
    callbacks = [
        EarlyStopping(patience=config.vae['patience']),
        LambdaCallback(on_epoch_begin=plot_samples),
        ModelCheckpoint(config.vae['ckpt_path'], verbose=1,
                        save_best_only=True, save_weights_only=True)
    ]

    # Load checkpoint if available
    if os.path.exists(config.vae['ckpt_path']):
        vae.load_weights(config.vae['ckpt_path'])
        log.info("Loaded VAE model weights from: %s", config.vae['ckpt_path'])

    # Fit VAE model!
    vae.fit(
        X_train, X_train,
        batch_size=config.vae['batch_size'],
        epochs=config.vae['epochs'],
        shuffle='batch',
        validation_split=0.2,
        callbacks=callbacks
    )


@cli.command()
@click.argument('path', type=click.Path(), required=True)
@click.option('-n', '--n_games', default=10000, help='Number of games to play (Default: 10000)')
# @click.option('-g', '--game_name', default='Pong-v0', help='OpenAI Gym game name (Default: Pong-v0)')
@click.option('-c', '--chunk_size', default=128, help='HDF5 chunk size (Default: 128)')
@click.option('-t', '--state_dtype', default='u1', help='Numpy data type of state (Default: uint8)')
# def record(path, n_games, game_name, chunk_size, state_dtype):
def record(path, n_games, chunk_size, state_dtype, game_name="Boxing-v0"):
    """Plays chosen game randomly and records transitions to hdf5 file in `PATH`."""

    # Create Gym environment, random agent and store to hdf5 callback
    env = hrl.create_gym(game_name)
    mind = RandomAgent(env.valid_actions)
    store_callback = StoreTransitions2Hdf5(
        env.valid_actions, STATE_SHAPE, path, chunk_size=chunk_size, dtype=state_dtype)

    # Resizes states to 64x64x3 with cropping
    vision = hrl.Vision(state_processor)

    # Play `N` random games and gather data as it goes
    hrl.loop(env, mind, vision, n_episodes=n_games, verbose=1, callbacks=[store_callback])


if __name__ == '__main__':
    cli()