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
import random
import os
import shutil
from datetime import datetime
import argparse


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


def set_global_seeds(seed: int) -> None:
    """
    Set seeds for reproducibility across Python, NumPy, TensorFlow, and Gym.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Load hyperparameters and environment from config.yaml
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

    seed = config.get("seed", 0)
    set_global_seeds(seed)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)

    # Save the config file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("experiments", exist_ok=True)
    config_copy_path = f"experiments/config_{timestamp}.yaml"
    shutil.copy("config.yaml", config_copy_path)
    logging.info(f"Saved config for this run to {config_copy_path}")

    # Log hyperparameters to TensorBoard at the start
    with summary_writer.as_default():
        tf.summary.text("hyperparam/seed", str(seed), step=0)
        for key, value in config["agent"].items():
            tf.summary.text(f"hyperparam/{key}", str(value), step=0)

    score_history: List[float] = []  # List to store episode scores
    np.random.seed(0)  # Set random seed for reproducibility

    # Main training loop for 1000 episodes
    for i in range(1000):
        observation, info = (
            env.reset()
        )  # Reset environment at the start of each episode
        done = False
        score = 0.0
        episode_critic_losses = []
        episode_actor_losses = []
        # Run one episode
        while not done:
            act = agent.choose_action(observation)  # Agent selects action
            new_state, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated  # Check if episode is done
            agent.remember(observation, act, reward, new_state, int(done))
            # Get losses from agent.learn()
            result = agent.learn()
            if result is not None:
                critic_loss, actor_loss = result
                episode_critic_losses.append(critic_loss)
                episode_actor_losses.append(actor_loss)
            score += reward  # Accumulate reward for this episode
            observation = new_state  # Move to next state
        score_history.append(score)  # Store episode score
        avg_critic_loss = (
            np.mean(episode_critic_losses) if episode_critic_losses else 0
        )
        avg_actor_loss = (
            np.mean(episode_actor_losses) if episode_actor_losses else 0
        )
        logging.info(
            "episode %d score %.2f trailing 100 games avg %.3f critic_loss %.4f actor_loss %.4f",
            i + 1,
            score,
            np.mean(score_history[-100:]),
            avg_critic_loss,
            avg_actor_loss,
        )
        # Log metrics to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar("Episode Reward", score, step=i)
            tf.summary.scalar("Average100", np.mean(score_history[-100:]), step=i)
            tf.summary.scalar("Critic Loss", avg_critic_loss, step=i)
            tf.summary.scalar("Actor Loss", avg_actor_loss, step=i)
    os.makedirs("results", exist_ok=True)
    # Save plot with hyperparameter info
    filename = (
        f"results/pendulum_tau{config['agent']['tau']}_bs{config['agent']['batch_size']}.png"
    )
    plotLearning(score_history, filename, window=100)
    # Save raw scores for later comparison
    score_file = (
        f"results/pendulum_tau{config['agent']['tau']}_bs{config['agent']['batch_size']}_scores.npy"
    )
    np.save(score_file, np.array(score_history))


if __name__ == "__main__":
    main()
