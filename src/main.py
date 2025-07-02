# Imports
import gym                      # OpenAI Gym for environment simulation
import numpy as np              # NumPy for numerical operations
from agent import Agent         # Import the Agent class from agent.py
from utils import plotLearning  # Utility function for plotting learning curves

if __name__ == '__main__':
    # Create the Pendulum-v1 environment from Gym
    env = gym.make('Pendulum-v1')
    # Initialize the DDPG agent with hyperparameters and environment details
    agent = Agent(
        alpha=0.0001,           # Learning rate for actor
        beta=0.001,             # Learning rate for critic
        input_dims=[3],         # State space dimensions
        tau=0.001,              # Soft update parameter for target networks
        env=env,                # The environment
        batch_size=64,          # Batch size for learning
        layer1_size=400,        # First hidden layer size
        layer2_size=300,        # Second hidden layer size
        n_actions=1             # Number of actions
    )

    score_history = []          # List to store episode scores
    np.random.seed(0)           # Set random seed for reproducibility

    # Main training loop for 1000 episodes
    for i in range(1000):
        observation, info = env.reset()   # Reset environment at the start of each episode
        done = False
        score = 0
        # Run one episode
        while not done:
            act = agent.choose_action(observation)                  # Agent selects action
            new_state, reward, terminated, truncated, info = env.step(act)  # Take action in environment
            done = terminated or truncated                         # Check if episode is done
            agent.remember(observation, act, reward, new_state, int(done))  # Store transition in replay buffer
            agent.learn()                                           # Agent learns from experience
            score += reward                                         # Accumulate reward for this episode
            observation = new_state                                 # Move to next state
        score_history.append(score)                                 # Store episode score
        print(
            'episode ', i,
            'score %.2f' % score,
            'trailing 100 games avg %.3f' % np.mean(score_history[-100:])
        )
    filename = 'pendulum.png'
    plotLearning(score_history, filename, window=100)               # Plot learning curve