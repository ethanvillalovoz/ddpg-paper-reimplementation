# Imports
import gym                      # OpenAI Gym for environment simulation
import numpy as np              # NumPy for numerical operations
from agent import Agent         # Import the Agent class from agent.py
from utils import plotLearning  # Utility function for plotting learning curves
import yaml                     # For loading configuration files
from typing import Any, Dict, List
import logging                  # Add logging import

def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path (str): Path to the YAML config file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main() -> None:
    """
    Main training loop for DDPG agent.
    Loads configuration, initializes environment and agent, and runs training episodes.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Load hyperparameters and environment from config.yaml
    config = load_config('config.yaml')
    env = gym.make(config['env'])
    agent = Agent(
        alpha=config['agent']['alpha'],
        beta=config['agent']['beta'],
        input_dims=config['agent']['input_dims'],
        tau=config['agent']['tau'],
        env=env,
        batch_size=config['agent']['batch_size'],
        layer1_size=config['agent']['layer1_size'],
        layer2_size=config['agent']['layer2_size'],
        n_actions=config['agent']['n_actions'],
        gamma=config['agent']['gamma'],
        max_size=config['agent']['max_size']
    )

    score_history: List[float] = []          # List to store episode scores
    np.random.seed(0)           # Set random seed for reproducibility

    # Main training loop for 1000 episodes
    for i in range(1000):
        observation, info = env.reset()   # Reset environment at the start of each episode
        done = False
        score = 0.0
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
        logging.info(
            'episode %d score %.2f trailing 100 games avg %.3f',
            i+1, score, np.mean(score_history[-100:])
        )
    filename = 'pendulum.png'
    plotLearning(score_history, filename, window=100)               # Plot learning curve

if __name__ == '__main__':
    main()