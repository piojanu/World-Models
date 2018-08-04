#!/usr/bin/env python3
import click
import third_party.humblerl as hrl

from third_party.humblerl.utils import RandomAgent
from third_party.humblerl.callbacks import StoreTransitions2Hdf5
from utils import pong_state_processor


@click.group()
def cli():
    pass


@cli.command()
@click.argument('path', type=click.Path(), required=True)
@click.option('-n', '--n_games', default=10000, help='Number of games to play (Default: 10000)')
#@click.option('-g', '--game_name', default='Pong-v0', help='OpenAI Gym game name (Default: Pong-v0)')
@click.option('-c', '--chunk_size', default=128, help='HDF5 chunk size (Default: 128)')
@click.option('-t', '--state_dtype', default='u1', help='Numpy data type of state (Default: uint8)')
#def record(path, n_games, game_name, chunk_size, state_dtype):
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
