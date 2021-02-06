"""
Script to play an Atari game with the model given in input.
The game is rendered while played
"""
import argparse
import numpy as np
import torch
from torch.autograd import Variable

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from a2c_pytorch.a2c import DeepLearningAgent


def reshape_observations_single_env(observations: np.array) -> Variable:
    """
    Reshape to channel first
    :param observations: (B, H, W, C) where B is the batch size
    :return: (B, C, H, W) observations
    """
    observations = observations.__array__().reshape(1, 84, 84, 4)
    return Variable(torch.from_numpy(observations.transpose((0, 3, 1, 2))).float() / 255.)


def main(arguments: argparse) -> None:
    """
    Play the game
    :param arguments: User input
    """
    print(f'Playing {args.game}')
    env = wrap_deepmind(make_atari(env_id=arguments.game), frame_stack=True)
    agent = DeepLearningAgent(observation_space=env.observation_space,
                              action_space=int(env.action_space.n),
                              n_envs=1,  # While playing, one environment at the time
                              n_steps=1,  # Dummy value, we are not training
                              model_path=arguments.model)
    # This is the current state (or observation)
    observations = reshape_observations_single_env(env.reset())
    actions = agent.get_action(observations, play=False)

    # Play maximum 10 games
    episodes = 10

    while episodes > 0:
        new_observations, rewards, done, info = env.step(actions[0])
        new_observations = reshape_observations_single_env(new_observations)
        actions = agent.get_action(new_observations, play=False)

        env.render()
        if done:
            episodes -= 1
            env.reset()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play A2C on Open AI Gym Atari.')
    parser.add_argument('--model', type=str, help='Existing model weights', default='saved_model.pytorch',
                        required=False)
    parser.add_argument('--game', type=str, help='Environment ID of the game', default="BreakoutNoFrameskip-v4",
                        required=False)

    args = parser.parse_args()
    main(args)
