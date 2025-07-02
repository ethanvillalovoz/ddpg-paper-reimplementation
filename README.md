# ddpg-paper-reimplementation
Reimplementation of "Continuous Control with Deep Reinforcement Learning" (ICLR 2016) using Tensorflow.

Using conda for my env:
conda create -n DDPG python=3.10

 Paper Summary
Title:
Continuous Control with Deep Reinforcement Learning
(Timothy P. Lillicrap et al., Google DeepMind, ICLR 2016)

🧠 What the Paper Is About
The paper introduces Deep Deterministic Policy Gradient (DDPG) — a model-free, off-policy reinforcement learning algorithm designed to solve continuous action space problems.

DDPG builds upon the Deterministic Policy Gradient (DPG) algorithm and incorporates innovations from Deep Q-Networks (DQN), such as:

Experience replay

Target networks for stability

DDPG enables stable and scalable learning of control policies in complex environments, including physical simulations and raw pixel inputs.

⚙️ Implementation Summary
✅ Algorithm Components
Actor-Critic Framework:

Actor: learns deterministic policy 
𝜇
(
𝑠
∣
𝜃
𝜇
)
μ(s∣θ 
μ
 )

Critic: learns action-value function 
𝑄
(
𝑠
,
𝑎
∣
𝜃
𝑄
)
Q(s,a∣θ 
Q
 )

Target Networks:
Slowly-updated copies of actor and critic used for computing stable targets.

Experience Replay Buffer:
Stores transitions 
(
𝑠
,
𝑎
,
𝑟
,
𝑠
′
)
(s,a,r,s 
′
 ) to sample uncorrelated minibatches.

Exploration Noise:
Added noise to actions using Ornstein-Uhlenbeck process for better exploration in physical environments.

Batch Normalization:
Normalizes inputs to stabilize training across varying scales.

🏗️ Neural Network Architecture
Actor and Critic: 2 fully connected hidden layers (400 + 300 units)

Pixel Input: 3 convolutional layers + 2 fully connected layers

Optimizer: Adam (1e-4 for actor, 1e-3 for critic)

Target update rate: 
𝜏
=
0.001
τ=0.001

Discount factor: 
𝛾
=
0.99
γ=0.99

Replay buffer size: 1 million

📊 Results Summary
🎯 Environments Tested
MuJoCo physical control tasks: e.g., Pendulum, Cartpole, Reacher, Cheetah, Walker2d, Gripper

TORCS driving simulator

Both low-dimensional state inputs and high-dimensional pixel inputs

📈 Key Findings
DDPG solved over 20 continuous control tasks with the same algorithm and hyperparameters.

In many tasks, DDPG matched or outperformed a planning-based baseline (iLQG with full access to environment dynamics).

Learning directly from pixels was successful in most environments.

Sample efficiency: policies learned in ≤ 2.5 million steps — far fewer than DQN on Atari (50M+ steps).

Stability was significantly improved using:

Replay buffer

Target networks

Batch normalization

🏆 Performance Example (Normalized Returns)
Environment	DDPG (Low-Dim)	DDPG (Pixels)	Planner (iLQG)
Pendulum	0.95	0.66	1.0
Cartpole	0.84	0.48	1.0
Cheetah	0.90	0.46	1.0
Walker2d	0.70	0.94	1.0
Torcs (raw score)	~1840	~1876	~1960

(0 = random policy, 1 = planning-based controller)

🔑 Key Contributions
First deep RL method to scale to high-dimensional continuous action spaces.

Stable and general method for learning from raw pixels without task-specific tuning.

Foundation for later algorithms like TD3 and SAC.

Ideas:
# Need a replay buffer class
# Need a class for a target Q network (function of state, action)
# We will use batch normalization
# The policy is deterministic, how to handle explore exploit?
# Deterministic policy means outputs the actual action insteat of a probability distribution
# Will need a way to bound the actions to the environment limits
# We gave two actor abd two critic networks, a target for each
# Updates are soft, according to theta_prime = tau * theta + (1 - tau) * theta_prime, with tau << 1
# The targert actor is just the evaluation actor plus some noise process
# They used Ornstein-Uhlenbeck noise for exploration