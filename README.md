# A2C Implementation in Pytorch

This package implements the A2C (Actor Critic) Reinforcement Learning approach to training Atari 2600 games.
It uses OpenAI Gym for the environments and Pytorch for the training process of the Neural network.

## Installation
TODO

## Training
To train, simply run the main file and let it train for aboud 1MM episodes. It should converge already after about 500K.
In my WSL environment it takes about 1 day using GPU.

```
python main.py
```

or, if you want to start from a pre-trained model

```
python main.py --model <PATH TO MODEL>
```

### Playing
To see the behavior of your trained environent run, after training

```
python play.py
```
Note: by default it uses saved_model.pytorch as policy
