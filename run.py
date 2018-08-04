#!/usr/bin/env python3
import click
import logging as log
import os.path
import third_party.humblerl as hrl

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import HDF5Matrix
from third_party.humblerl.utils import RandomAgent
from third_party.humblerl.callbacks import StoreTransitions2Hdf5
from utils import Config, pong_state_processor
from vision import build_vae_model


@click.group()
@click.pass_context
@click.option('-c', '--config_path', type=click.Path(exists=True), default="config.json",
              help="Path to configuration file (Default: config.json)")
@click.option('--debug/--no-debug', default=False, help="Enable debug logging (Default: False)")
def cli(ctx, config_path, debug):
    # Get and set up logger level and formatter
    log.basicConfig(level=log.DEBUG if debug else log.INFO, format="[%(levelname)s]: %(message)s")

    # Load configuration from .json file into ctx object
    ctx.obj = Config(config_path, debug)


@cli.command()
@click.pass_context
@click.argument('path', type=click.Path(exists=True), required=True)
def train_vae(ctx, path):
    """Train VAE model as specified in .json config with data at `PATH`."""

    config = ctx.obj

    # Get training data
    X_train = HDF5Matrix(path, 'states', normalizer=lambda x: x / 255.)
    y_train = HDF5Matrix(path, 'next_states', normalizer=lambda x: x / 255.)

    # Build VAE model
    vae, _, _ = build_vae_model(config.vae)

    # Initialize callbacks
    callbacks = [
        EarlyStopping(patience=config.vae['patience']),
        ModelCheckpoint(config.vae['ckpt_path'], verbose=1,
                        save_best_only=True, save_weights_only=True)
    ]

    # Load checkpoint if available
    if os.path.exists(config.vae['ckpt_path']):
        vae.load_weights(config.vae['ckpt_path'])
        log.info("Loaded VAE model weights from: %s", config.vae['ckpt_path'])

    # Fit VAE model!
    vae.fit(
        X_train, y_train,
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
def record(path, n_games, chunk_size, state_dtype, game_name="Pong-v0"):
    """Plays chosen game randomly and records transitions to hdf5 file in `PATH`."""

    # Create Gym environment, random agent and store to hdf5 callback
    env = hrl.create_gym(game_name)
    mind = RandomAgent(env.valid_actions)
    store_callback = StoreTransitions2Hdf5(
        env.valid_actions, (64, 64, 3), path, chunk_size=chunk_size, dtype=state_dtype)

    # Resizes states to 64x64 with cropping
    vision = hrl.Vision(pong_state_processor)

    # Play `N` random games and gather data as it goes
    hrl.loop(env, mind, vision, n_episodes=n_games, verbose=1, callbacks=[store_callback])


if __name__ == '__main__':
    cli()
