# A2C Implementation in Pytorch

This package implements the A2C (Actor Critic) Reinforcement Learning approach to training Atari 2600 games.
It uses OpenAI Gym for the environments and Pytorch for the training process of the Neural network.

## Installation
Create an environment with your favorite tool. (I use anaconda)

```
conda create -n a2c
conda activate a2c
conda install pip
pip install -r requirements.txt
pip install -e .
```

Note: this will install some library which requires building and/or other system libraries which you might need to install manually (using apt)
The code has been tested in an Ubuntu distribution running on Windows over WSL.

Note: It uses a port of OpenAI baselines from which I have extracted the wrap environments functionalities to avoid adding a dependency to tensorfow

## Training
To train, simply run the main file and let it train for aboud 1MM episodes. It should converge already after about 500K.
In my WSL environment it takes about 1 day using GPU (which is used by defaul)

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
