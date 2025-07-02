import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    plt.figure(figsize=(12, 8))
    for score_file in glob.glob("results/*_scores.npy"):
        scores = np.load(score_file)
        label = os.path.splitext(os.path.basename(score_file))[0]
        plt.plot(scores, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("DDPG Hyperparameter Sweep Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
