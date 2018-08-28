# Deep Q-Learning agent in Bananas Environment


## Overview

This repository contains code for training and running a reinforcement learning agent which has to learn to interact with a 3D environment in which it has to avoid poisonous purple bananas, and consume healthy yellow bananas.


The agent starts out with no prior knowledge of the environment, what its goal is, what its actions are, or what effect those actions have in the environment.

Through rewards and punishments, the agent has to learn over time to navigate the 3D environment, avoid the poisonous bananas and seek out the healthy bananas.

The agent inplemented here makes use of the Deep Q-learning algorithm, using a neural network with two hidden layers to  create a representation  of the environment, and appropriate actions to take under different states.

You can train the agent from scratch yourself, or make use of a pretrained agent using the saved snapshot of the agent provided in the `snapshot.pth` file.

Below is a video of the trained agent interacting with the world.


<iframe width="560" height="315" src="https://www.youtube.com/embed/cSQVqcWtz2o" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>


## Environment Description

**States :**

- A vector with 37 items.
- Contains information about the agent's velocity
- Also contains a ray-based perception of objects around the agent's forward direction.

**Actions :**

- 0: forward
- 1: backward
- 2: left
- 3: right

**Rewards :**

- **+1 :** for collecting a yellow banana
- **-1** for collecting a blue banana

**Goal:**

- Collect as many yellow bananas as possible, while avoiding the blue ones.
- It is considered solved if it gets an average score of 13 over a rolling window of 100 consecutice episodes of game play.


## File Structure

This repsitory contains the following files:

- **model.py** Contains the neural network that controls the behaviour of the agent. It is implemented in pytorch.
- **agent.py** Contains the `Agent` class, which wraps around the neural network model, and contains methods for training the model, and chosing actions based on its internal representation.
- **environment.py**
    - Contains `UnityEnvWrapper` which creates an object, which acts as an interface for interacting with the Unity environment, using a similar simple API (but not quite the same) as an OpenAI Gym environment object.
    - It also contains a `Interact` class, that glues an environment, and an agent object and allows you to make the two interact during training and evaluation.
- **banana_train.py**
    - The script that will train the agent.
    - Once trained succesfully, it will create a snapshot file of the trained model in the same directory.
- **banana_play.py**
    - A script that will play a round of the environment using a pre-trained agent.
    - NOTE: this requires a snapshot file to have been created and stored in the same directory.
- **report.md**
    - report that outlines the game, and the structure of the neural network used.


## Prerequisites

This repository requires python 3.6, along with the following python libraries.

```
numpy
matplotlib
pytorch
```

Which can be installed using pip:
```
pip install -U numpy
pip install -U matplotlib
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
```

It also requires a modified version of the Unity Unity Machine Learning Agents python library, which can be set up along with other dependencies by running:

```sh
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

It also makes use of a compiled Unity Environment for the Bananas game.

Simply download the zip file for your operating system using one of the following links, and then extract the contents of the zip file into the same working directory as this repository.


[Linux (32 and 64 bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
[Linux (NO GUI)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip)
[Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
[Windows 32 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
[WIndows 64 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


eg, on linux, you can download and extract the GUI version by running:

```sh
wget -c https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
unzip Banana_Linux.zip
```

## Setup

Open up the `banana_train.py` file, and modify the line that says:

```
env = UnityEnvWrapper("Banana_Linux/Banana.x86_64", seed=seed)
```

Change it so it uses the file path to the Unity Bananas environment you donwloaded in the previous step.

Now do the same thing for the `banana_play.py` file.


## Training

Run the `banana_train.py` file in the terminal, eg by running:

```sh
python banana_train.py
```

And you will see feedback on the terminal as it trains, similar to the following, and it will save a snapshot of the trained agent to `snapshot.pth` once it has trained succesfully.

```
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :

Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , ,

TRAINING AGENT

Episode 100	Average Score: 0.572
Episode 200	Average Score: 2.57
Episode 300	Average Score: 6.70
Episode 400	Average Score: 9.97
Episode 500	Average Score: 12.46
Episode 519	Average Score: 13.02
Environment solved in 419 episodes!	Average Score: 13.02
```

## Running a trained agent

To run a trained agent, using the weights saved in `snapshot.pth`

```
python banana_run.py
```

## Credits

Much of the codebase in this repository is based on the starter code provided by the [Udacity Deep Reinforcement Learning Nanodegree](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation). 
