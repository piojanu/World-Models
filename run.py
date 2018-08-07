#!/usr/bin/env python3
import click
import h5py as h5
import datetime as dt
import logging as log
import numpy as np
import os
import third_party.humblerl as hrl

from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from third_party.humblerl.utils import RandomAgent
from third_party.humblerl.callbacks import StoreTransitions2Hdf5
from utils import Config, HDF5DataGenerator, boxing_state_processor as state_processor
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

    # Get dataset length and eight examples to evaluate VAE on
    with h5.File(path, 'r') as hfile:
        n_transitions = hfile.attrs['N_TRANSITIONS']
        X_eval = hfile['states'][:8] / 255.

    # Get training data
    train_gen = HDF5DataGenerator(path, 'states', 'states', batch_size=config.vae['batch_size'],
                                  end=int(n_transitions * 0.8),
                                  preprocess_fn=lambda X, y: (X / 255., y / 255.))
    val_gen = HDF5DataGenerator(path, 'states', 'states', batch_size=config.vae['batch_size'],
                                start=int(n_transitions * 0.8),
                                preprocess_fn=lambda X, y: (X / 255., y / 255.))

    # Build VAE model
    vae, _, _ = build_vae_model(config.vae)

    # If render features enabled...
    if config.allow_render:
        # ...plot first eight training examples with VAE reconstructions
        # at the beginning of every epoch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        # Check if destination dir exists
        plots_dir = os.path.join(config.vae['logs_dir'], "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Evaluate VAE at the end of epoch
        def plot_samples(epoch, logs):
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
            plt.close()
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
    vae.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        epochs=config.vae['epochs'],
        use_multiprocessing=True,
        # TODO:  Make generator multi-thread.
        # NOTE:  There is no need for more then one workers, we are disk IO bound (I suppose ...)
        # NOTE2: h5py from conda should be threadsafe... but it apparently isn't and raises
        #        `OSError: Can't read data (wrong B-tree signature)` sporadically if `workers` = 1
        #        and always if `workers` > 1. That's why this generator needs to run in main thread
        #        (`workers` = 0).
        workers=0,
        max_queue_size=100,
        callbacks=callbacks
    )


@cli.command()
@click.pass_context
@click.argument('model_path', type=click.Path(exists=True), required=True)
@click.argument('in_path', type=click.Path(exists=True), required=True)
@click.argument('out_path', type=click.Path(), required=True)
def preproc_vae(ctx, model_path, in_path, out_path):
    """Preprocess data at 'IN_PATH' using VAE model at `MODEL_PATH` so it can be used in Memory 
    module training. Save preprocessed data at `OUT_PATH` as numpy array."""

    config = ctx.obj

    # Get training data
    s_gen = HDF5DataGenerator(in_path, 'states', 'states', batch_size=config.vae['batch_size'],
                              preprocess_fn=lambda X, y: (X / 255., y / 255.))
    ns_gen = HDF5DataGenerator(in_path, 'next_states', 'states', batch_size=config.vae['batch_size'],
                               preprocess_fn=lambda X, y: (X / 255., y / 255.))

    # Build VAE model
    vae, encoder, _ = build_vae_model(config.vae)

    # Load checkpoint if available
    if os.path.exists(config.vae['ckpt_path']):
        vae.load_weights(config.vae['ckpt_path'])
        log.info("Loaded VAE model weights from: %s", config.vae['ckpt_path'])
    else:
        raise ValueError("VAE model weights from \"{}\" path doesn't exist!".format(model_path))

    # Infer VAE model!
    log.info("Encode states...")
    encoded_states = encoder.predict_generator(
        generator=s_gen,
        use_multiprocessing=True,
        workers=0,
        max_queue_size=100,
        verbose=1
    )[0]

    log.info("Encode next states...")
    encoded_next_states = encoder.predict_generator(
        generator=ns_gen,
        use_multiprocessing=True,
        workers=0,
        max_queue_size=100,
        verbose=1
    )[0]

    # Get actions
    log.info("Get actions...")
    with h5.File(in_path, 'r') as hfile:
        transitions = hfile['transitions']
        actions = transitions[:, 1]

    # Save dataset
    np.savez(out_path, states=encoded_states, actions=actions, next_states=encoded_next_states)


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
