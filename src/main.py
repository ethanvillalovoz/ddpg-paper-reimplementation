# Imports
import gym  # OpenAI Gym for environment simulation
import numpy as np  # NumPy for numerical operations
from agent import Agent  # Import the Agent class from agent.py
from utils import plotLearning  # Utility function for plotting learning curves
import yaml  # For loading configuration files
from typing import Any, Dict, List
import logging  # Add logging import
from env_wrappers import NormalizedEnv  # Add this import
import tensorflow as tf  # For TensorBoard experiment tracking


def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path (str): Path to the YAML config file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Main training loop for DDPG agent.
    Loads configuration, initializes environment and agent, and runs training episodes.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Load hyperparameters and environment from config.yaml
    config = load_config("config.yaml")
    env = gym.make(config["env"])
    env = NormalizedEnv(env)  # Wrap the environment for normalization
    agent = Agent(
        alpha=config["agent"]["alpha"],
        beta=config["agent"]["beta"],
        input_dims=config["agent"]["input_dims"],
        tau=config["agent"]["tau"],
        env=env,
        batch_size=config["agent"]["batch_size"],
        layer1_size=config["agent"]["layer1_size"],
        layer2_size=config["agent"]["layer2_size"],
        n_actions=config["agent"]["n_actions"],
        gamma=config["agent"]["gamma"],
        max_size=config["agent"]["max_size"],
        actor_path=config.get("actor_path", "actor.h5"),
        critic_path=config.get("critic_path", "critic.h5"),
        target_actor_path=config.get("target_actor_path", "target_actor.h5"),
        target_critic_path=config.get("target_critic_path", "target_critic.h5"),
    )

    # Create a summary writer for TensorBoard
    summary_writer = tf.summary.create_file_writer("runs/ddpg_experiment")

    score_history: List[float] = []  # List to store episode scores
    np.random.seed(0)  # Set random seed for reproducibility

    # Main training loop for 1000 episodes
    for i in range(1000):
        observation, info = (
            env.reset()
        )  # Reset environment at the start of each episode
        done = False
        score = 0.0
        # Run one episode
        while not done:
            act = agent.choose_action(observation)  # Agent selects action
            new_state, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated  # Check if episode is done
            agent.remember(observation, act, reward, new_state, int(done))
            agent.learn()  # Agent learns from experience
            score += reward  # Accumulate reward for this episode
            observation = new_state  # Move to next state
        score_history.append(score)  # Store episode score
        logging.info(
            "episode %d score %.2f trailing 100 games avg %.3f",
            i + 1,
            score,
            np.mean(score_history[-100:]),
        )
        # Log metrics to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar("Episode Reward", score, step=i)
            tf.summary.scalar(
                "Average100", np.mean(score_history[-100:]), step=i
            )
    filename = "pendulum.png"
    plotLearning(score_history, filename, window=100)  # Plot learning curve


if __name__ == "__main__":
    main()
