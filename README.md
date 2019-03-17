[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

![Trained Agent][image1]

### Overview

This is the 1st project for the Udacity Deep Reinforcement Learning Nanodegree program. The goal of the project is to learn implementing a deep neural network as Q function approximator to perform a discrete action task. In this project, the Navigation environemnt is utilized.

In this environment, we will train an agent to navigate (and collect bananas!) in a large, square world. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### File Instruction
_**Navigation.ipynb**_: This is the python jupyter notebook file which performs the followings:
1. Start the environment and examine the state and action space
2. Take random actions in the environement to become familar with the environment API
3. Train the agent using deep neural network to solve the environement

_**dqn_agent.py**_: The python code to implement the [deep Q learning (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). Several extensions have been done to improve the model performance:
- **Double DQN:** Deep Q-Learning tends to overestimate action values. [Double Q-Learning](https://arxiv.org/abs/1509.06461) has been shown to work well in practice to help with this. 
- **Prioritized Experience Replay:** Deep Q-Learning samples experience transitions uniformly from a replay memory. 
[Prioritized experienced replay](https://arxiv.org/abs/1511.05952) is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability.

_**model.py**_: The python code to configure the neural network.

_**checkpoint.pth**_: The network model weights are saved in the checkpoint file

### Requirements
1. Install Anaconda distribution of python3

2. Install PyTorch, Jupyter Notebook in the Python3 environment

3. Download the environment from one of the links below and place it in the same directory folder. Only select the environment that matches the operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

4. Open Jupyter Notebook and run the `Navigation.ipynb` file to train the agent. 

5. To watch the agents to play, load the model weights from checkpiont _.pth_ files by executing all the notebook cells except **Training** session.
