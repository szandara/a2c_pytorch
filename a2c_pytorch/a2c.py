"""
Main A2C implementation. A2C follows the original
implementation of A3C but removes the asynchronous behavior.

Each batch is calculated by running all environments at the same time
and batching the data in a single batch for training.

https://paperswithcode.com/paper/asynchronous-methods-for-deep-reinforcement
https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
"""
import time
from collections import namedtuple
from typing import Tuple, List
import gym
import torch

import torch.nn.functional as F
import numpy as np

from a2c_pytorch.network import CNN
from torch.distributions import Categorical

MODEL_PATH = 'saved_model.pytorch'

# Contains 1 step of data for each environemnt
Episode = namedtuple('Episode', ['rewards', 'masks', 'actions', 'state'])
# Contains the batch data used for training
PreparedEpisode = namedtuple('Episode', ['actions', 'returns', 'state'])


class DeepLearningAgent:
    """
    Deep learning Q policy for Atari OpenAI gym
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: int,
                 n_envs: int,
                 n_steps: int,
                 ent_coeff: float = 0.01,
                 value_coeff: float = 0.5,
                 gradient_clip_max: float = 0.5,
                 gamma: float = 0.99,
                 learning_rate: float = 3e-4,
                 model_path: str = None,
                 use_cpu: bool = False):
        """
        :param observation_space: size of the input data
        :param action_space: number of possible actions
        :param n_envs: number of parallel environments
        :param n_steps: number of steps per environment per batch size
        :param ent_coeff: entropy coefficient for the loss function
        :param model_path: existing model which can be used as starting point for training or replaying
        """
        self._image_size = observation_space.shape

        if use_cpu:
            self._device = torch.device("cpu:0")
        else:
            self._device = torch.device("cuda:0")

        self._gamma = gamma
        self._ent_coeff = ent_coeff
        self._value_coeff = value_coeff
        self._gradient_clip_max = gradient_clip_max

        self._batch_train_n = 0
        self._action_space = action_space
        self._actions = list(range(self._action_space))
        self._n_steps = n_steps
        self._num_envs = n_envs
        self._batch_size = self._num_envs * n_steps
        print(f'Batch Size = {self._batch_size}')

        self._epsilon = 1

        self._model = CNN(num_actions=self._action_space).to(self._device)
        if model_path:
            self._model.load_state_dict(torch.load(model_path))

        # Optimizer
        self._optimizer = torch.optim.RMSprop(self._model.parameters(),
                                              lr=learning_rate)

        # Training data, init to empty and format in reset()
        self._last_training_time = time.time()
        self._total_rewards = 0
        self._total_loss = 0

        # Replay data
        self._steps = []

    def _predict(self, state: torch.Tensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Run the prediction on the input image
        :param state: Input image in Tensor format (batch_size x channels x W x H)
        :return: output of the fully connected network flattened
                 batch_size x convolution_output
        """
        x_tensor = state.to(self._device)
        value, policy = self._model(x_tensor)
        return value, policy

    @classmethod
    def _sample_from_policy(cls, policy: torch.FloatTensor) -> np.array:
        """
        Sample an action from the policy using the distribution
        returned by the network (Actor)
        :param policy: Tensor containing the policy (batch_size x num_actions)
        :return: array of actions sized batch_size
        """
        softmax = F.softmax(policy, dim=1)
        return torch.multinomial(softmax,
                                 num_samples=1).data

    def get_action(self, state: np.array, play: bool = False) -> np.array:
        """
        Return an action for the given input observation (state)
        :param state: current visual observation from the game
        :param play: Use probability distribution or simply get the argmax?
        :return: arrays of actions of size batch_size
        """

        # Disable the gradients when getting an action for speed
        with torch.no_grad():
            if play:
                values, policy = self._predict(state)
                action = np.array([torch.argmax(policy).item()])
            else:
                values, policy = self._predict(state)
                action = self._sample_from_policy(policy.detach())
        return action

    def _train_step(self,
                    steps: list,
                    s_next: np.array) -> torch.Tensor:
        """
        Updates the weights of the network with the data batch
        :param steps: It's a list of tuples
        :param s_next: last observation
        :return: latest loss from the loss function
        """

        # Collect the latest values
        s_next_tensor = s_next.to(self._device)
        next_values, _ = self._model(s_next_tensor)

        prepared_data = self._calculate_rewards_and_stack_data(steps=steps,
                                                               last_values=next_values)

        batch_actions = prepared_data.actions
        returns = prepared_data.returns
        states = prepared_data.state

        self._optimizer.zero_grad()

        # Remove additional dimensions to avoid broadcasting errors
        returns = returns.squeeze()
        batch_actions = batch_actions.squeeze()

        # Actor/Critic values of the given batch
        batch_values, batch_policies = self._model(states.to(self._device))
        batch_values = batch_values.squeeze()

        # Calculate the  loss as in https://arxiv.org/pdf/1602.01783.pdf
        # Use the same update as A3C.
        batch_policies_distribution = Categorical(batch_policies)
        action_probabilities = batch_policies_distribution.log_prob(batch_actions)

        # Entropy loss using the distribution from the policies of the chosen actions
        entr_loss = batch_policies_distribution.entropy().mean()

        # Policy loss
        policy_loss = -(action_probabilities * (returns - batch_values).detach()).mean()

        # Value loss
        value_loss = F.mse_loss(returns, batch_values)
        ac_loss = policy_loss - self._ent_coeff * entr_loss + self._value_coeff * value_loss

        # Calculate the gradients
        ac_loss.backward()

        # Clip the gradients to avoid exploding weights
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._gradient_clip_max)

        # Update the weights
        self._optimizer.step()
        return ac_loss.detach()

    def train(self, s: np.array,
              a: np.array,
              r: np.array,
              s_next: np.array,
              done: np.array,
              step: int) -> None:
        """
        Collects the episode data from the environment and run
        the training update when the batch size has been collected

        Furthermore it prints information about the training progress

        :param s: current state (observations) size (num_env x observation_size)
        :param a: current action (num_env x 1)
        :param r: reward from current action a in state s (num_env x 1)
        :param s_next: state result of action a in status s (num_env x 1)
        :param done: True if the episode is over (num_env x observation_size)
        :param step: step index in frames
        """

        rewards = torch.from_numpy(r).float().unsqueeze(1).to(self._device)
        masks = (1. - torch.from_numpy(np.array(done, dtype=np.float32))).unsqueeze(1).to(self._device)

        self._steps.append(Episode(rewards=rewards,
                                   masks=masks,
                                   actions=a,
                                   state=s))
        self._total_rewards += rewards.sum()

        if self._n_steps - 1 == step:
            self._batch_train_n += 1

            # Train the network
            loss = self._train_step(steps=self._steps, s_next=s_next)

            if self._batch_train_n % 100 == 0:
                new_time = time.time()
                print(f'Episode {self._batch_train_n}')
                print(f'Average Loss {self._total_loss / 100}')
                print(f'Average Rewards {self._total_rewards / 100}')
                print(f'Total Rewards last 100 steps {self._total_rewards}')
                print(f'Training time of 100 batches {new_time - self._last_training_time} ms')
                print(f'----------------------------')
                self._last_training_time = new_time

                # Save the current training status
                torch.save(self._model.state_dict(), MODEL_PATH)
                self._total_rewards = 0
                self._total_loss = 0
            else:
                self._total_loss += loss

            self._steps = []

    def _calculate_rewards_and_stack_data(self, steps: List[Episode],
                                          last_values: torch.Tensor) -> PreparedEpisode:
        """
        Calculate the returns of the actions and re-aggregate the data.
        Input data is in format

        data[step1, env1]
        data[step1, env2]
        ...

        and converted to a Tensor of format batch_size x data_size

        [env1, step1]
        [env1, step2]
        ...
        [envN, stepN-1]
        [envN, stepN]

        :param steps: list of Episode tuples containing episode data
        :param last_values: value of the last episode
        :return: Tuple of tensors containing data
                [0] Action tensor
                [1] Return tensor
                [2] Observation tensor
        """
        returns = last_values

        out = [None] * (len(steps) - 1)

        # calculate the returns and stack the data per environment
        for t in reversed(range(len(steps) - 1)):
            rewards, masks, actions, state = steps[t]

            returns = rewards + returns * self._gamma * masks
            out[t] = actions, returns, state

        # return data as batched Tensors
        actions, returns, state = map(lambda x: torch.cat(x, 0), zip(*out))
        return PreparedEpisode(actions=actions,
                               returns=returns,
                               state=state)
