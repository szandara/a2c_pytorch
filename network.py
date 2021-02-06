"""
Neural network used to train the Atari environments.
It's a 2 layer convolutional network with two heads, commonly used for
Atari games.
"""
import numpy as np
import torch

from torch import nn
from typing import Tuple


def ortho_weights(shape, scale=1.):
    """ PyTorch port of ortho_init from baselines.a2c.utils """
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError

    a = np.random.normal(0., 1., flat_shape)

    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))


def atari_initializer(module: nn.Module) -> None:
    """
    Initializes a network layer with orthogonal weights.
    https://smerity.com/articles/2016/orthogonal_init.html
    Orthogonal weights prevents the layers from vanishing or exploding
    by ensuring that the Eigen values are all 1.
    :param module: layer of a Pytorch network
    """
    classname = module.__class__.__name__

    if classname == 'Linear' or classname == 'Conv2d':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()


class CNN(nn.Module):
    HIDDEN_SIZE = 512  # Arbitrary fully connected network size
    CHANNELS = 4  # Channels encode sequences of frames rather than RGB informations.

    # In this case, each image contains 4 frames encoded as 4 changes.

    def __init__(self, num_actions):
        """
        Actor-critic network for Atari 2600 games as defined in the DQN paper.
        https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        :param num_actions: Number of possible actions of the game
        """
        super().__init__()

        self.num_actions = num_actions

        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.CHANNELS, out_channels=32, kernel_size=8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                  nn.ReLU())
        convolution_output = 64 * 7 * 7

        self.fc = nn.Sequential(nn.Linear(in_features=convolution_output, out_features=self.HIDDEN_SIZE),
                                nn.ReLU())

        # Actor value
        self.pi = nn.Linear(self.HIDDEN_SIZE, num_actions)
        # Critic value
        self.v = nn.Linear(self.HIDDEN_SIZE, 1)

        # parameter initialization
        self.apply(atari_initializer)

    def forward(self, conv_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network which runs a forward pass
        to the main network with the input images and then
        calls the actor/critic networks
        :param conv_in: convolutional input, shaped [batch_size x 4 x 84 x 84]
        :return: Tuple of values
                pi = (Actor) action probability logits, shaped [batch_size x self.num_actions]
                v = (Critic) value predictions, shaped [batch_size x 1]
        """
        fc_out = self.body(conv_in)

        pi_out = self.pi(fc_out)
        v_out = self.v(fc_out)

        return v_out, pi_out

    def body(self, conv_in: torch.Tensor) -> torch.Tensor:
        """
        Main netork body which contains a 3 layer convolution
        and a fully connected network pass
        :param conv_in:
        :return: output of the fully connected network flattened
                 batch_size x convolution_output
        """
        batch_size = conv_in.size()[0]

        conv_out = self.conv(conv_in).reshape(batch_size, -1)
        fc_out = self.fc(conv_out)
        return fc_out
