"""
Main training loop. It runs the game for a given set of episodes
and meanwhile trains the network using the rewards.

It performs an off-policy A2C training with multiple
simultaneous games playing in separate environments.

References:
    - https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl3/a2c
    - https://github.com/lnpalmer/A2C
"""
import time
import argparse

from baselines.common.vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from a2c_pytorch.a2c import DeepLearningAgent
from a2c_pytorch.utils import reshape_observations

SEED = 0
EPISODES = int(1e6)


def make_env(env_id, rank):
    """
    Returns a function which is lazily used to create an
    atary playable environment with the right id and randomizer
    :param env_id: type of the environmet (game)
    :param rank: index of the environment used for random seeding
    :return:
    """

    def f():
        env_a = make_atari(env_id)
        env_a.seed(SEED + rank)
        # Setup the environment using the deepmind standards
        # Each observation contains 4 stacked frames
        env_a = wrap_deepmind(env_a, frame_stack=True)

        return env_a

    return f


def main(arguments: argparse) -> None:
    """
    Main training loop.
    :param arguments: User input
    :return:
    """
    n_steps = arguments.steps
    n_agents = arguments.envs

    print(f'Training {args.game}')
    print(f'Number of concurrent environments {args.envs}')
    print(f'Number of steps per batch {args.steps}')

    if arguments.model:
        print(f'Using existing model {arguments.model}')

    env = SubprocVecEnv([make_env(env_id=arguments.game, rank=i) for i in range(n_agents)])
    agent = DeepLearningAgent(observation_space=env.observation_space,
                              action_space=int(env.action_space.n),
                              n_envs=n_agents,
                              n_steps=n_steps,
                              model_path=arguments.model)

    # This is the current state (or observation)
    observations = reshape_observations(env.reset())
    actions = agent.get_action(observations)
    initial_training_time = time.time()

    for ep in range(EPISODES):
        # Reset the frame counter each time the batch size is complete
        for i in range(n_steps):
            new_observations, rewards, done, info = env.step(actions.cpu().numpy())
            new_observations = reshape_observations(new_observations)

            agent.train(s=observations,
                        r=rewards,
                        s_next=new_observations,
                        a=actions,
                        done=done,
                        step=i)

            actions = agent.get_action(new_observations)
            observations = new_observations

        if ep % 100 == 0:
            fps = ((ep + 1) * n_steps * n_agents) / (time.time() - initial_training_time)
            print(f'FPS {fps}')

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train A2C on Open AI Gym Atari.')
    parser.add_argument('--model', type=str, help='Existing model weights', default=None, required=False)
    parser.add_argument('--envs', type=int, help='Number of concurrent environments', default=16,
                        required=False)
    parser.add_argument('--steps', type=int, help='Number of steps per batch_size', default=5,
                        required=False)
    parser.add_argument('--game', type=str, help='Environment ID of the game', default="BreakoutNoFrameskip-v4",
                        required=False)

    args = parser.parse_args()
    main(args)
