import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plotLearning(
    scores: List[float],
    filename: str,
    window: int = 100,
    critic_losses: List[float] = None,
    actor_losses: List[float] = None,
) -> None:
    """
    Plots the running average of scores over a specified window and saves the plot
    to a file.

    Args:
        scores (list or np.ndarray): List of episode scores.
        filename (str): Path to save the plot image.
        window (int): Window size for running average.
    """
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window) : (t + 1)])
    plt.figure(figsize=(12, 8))
    plt.plot(running_avg, label=f"Running Avg (window={window})")
    if critic_losses is not None:
        plt.plot(critic_losses, label="Critic Loss (avg/ep)", alpha=0.7)
    if actor_losses is not None:
        plt.plot(actor_losses, label="Actor Loss (avg/ep)", alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Score / Loss")
    plt.title(f"Running average of previous {window} scores and losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
