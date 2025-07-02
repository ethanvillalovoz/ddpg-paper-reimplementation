import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plotLearning(scores: List[float], filename: str, window: int = 100) -> None:
    """
    Plots the running average of scores over a specified window and saves the plot to a file.

    Args:
        scores (list or np.ndarray): List of episode scores.
        filename (str): Path to save the plot image.
        window (int): Window size for running average.
    """
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    plt.figure(figsize=(10, 6))
    plt.plot(running_avg, label=f'Running Avg (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'Running average of previous {window} scores')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()