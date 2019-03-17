# Project 2: Continuous Control task

### Deep _Q_ Network (_DQN_)

In this excise, we learn how to use [_DQN_ algorithm](http://files.davidqiu.com//research/nature14236.pdf) to solve the reinforcement learning task with high dimensional
state space. 
The _DQN_ utilizes the deep neural network to approximate the action-value function _Q(s,a)_ and therefore select
the actions in a fashion that maximizes the function value. In the algorithm, a replay buffer is implemented to randomize the
sequential experience data (_S, A, R, S'_) to remove the correlation in the observation sequence and therefore improve the model stability.
Further, the update of the _Q_ target values are done in a periodical manner that reduces the correlations of 
the estimate and the target _Q_ values.  

The algorithm details is seen **Algorithm 1** figure below. First it initializes an agent with **estimated local** _Q(s,a|&theta;) network_  
and **target** _Q(s,a|&theta;') network_ and _prioritized replay buffer_. 
Here the [prioritized experienced replay](https://arxiv.org/abs/1511.05952) is a variant of replay buffer, which makes the learning 
process more efficient and effective by setting higher frequency for samples associated with high temporal difference (TD) error. 
The agent selects and executes actions according to an &epsilon;-greedy policy based on estimated _Q_. 
By excecuting the action, the reward and the new state is observed, and then these experiences are added to the prioritized replay buffer.
Next it comes to learning the deep network model weight. Here a separate _Q_ network is maintained for the **target** _Q_ values. 
Specifically, every update only a small percentage (_&tau;_) of the **target** network weights are updated by copying from the **estimated local** _Q_ network
model weights. Note that, in standard _DQN_ the target _Q_ value is _R+&gamma;maxQ(S,a; &theta;')_ where the max operator uses the same
values both to select and to evaluate an action, which makes it more likely to overestimate values. To decouple the selection from the 
evaluation, here [Double Q-Learning](https://arxiv.org/abs/1509.06461) is implemented where the target values are 
_R+&gamma;Q(S, argmaxQ(S,a; &theta;); &theta;')_. Note that the selection of the action in the _argmax_ is still due to
the online local network weights **&theta;**; whereas the target value evaluation is done through a separate weight set **&theta;'**.  

![DQN algorithm](https://github.com/ChaoLiRV/Udacity_DRL_Navigation/blob/master/dqn_algo.png)

 ### Implementation details
_**deep neural network:**_ It takes the state observation as input and returns the action as the output. 
The neural network has two hidden layers with 32 and 16 nodes. 
Both layers use _ReLU_ activation function. The output layer uses linear activation function

In the algorithm, the buffer size is set as _1e+5_ and batch sample size for training is _64_. The hyperparameters for prioritized replay 
buffer are _&alpha;=0.6, &beta;=0.4_. The &epsilon; value is set to _1.0_ at a higher level to allow for full exploration 
at beginning, and gradually decay to a lower level _0.01_ with decay rate of _0.99_ per time-step to focus more on exploitation. 
The reward discounting factor _&gamma;=0.99_ and learning rate _1e-3_.The update parameter _&tau;_
for the target is _1e-2_

### Score plot
The environment is solved at **371** episodes. Refer to the plot below to see how the average scores evolve as a function of episodes.

![score plot](https://github.com/ChaoLiRV/Udacity_DRL_Navigation/blob/master/score_plot.png)  

### Future work
- Clip the TD error term to be between -1 and 1 in order to further improve the stability of the algorithm
- Implement Batch Normalization to process the input to improve the learning efficiency 
- Combine the following extensions to the DQN algorithms, like what is done by [Rainbow](https://arxiv.org/abs/1710.02298)
    - [**Dueling DQN**](https://arxiv.org/abs/1511.06581)
    - [**multi-step bootstrap targets**](https://arxiv.org/abs/1602.01783)
    - [**Distributional DQN**](https://arxiv.org/abs/1707.06887)
    - [**Noisy DQN**](https://arxiv.org/abs/1707.06887)