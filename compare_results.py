import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    Load and plot score files from the results directory for comparison.
    Each *_scores.npy file is plotted as a separate curve.
    """
    plt.figure(figsize=(12, 8))  # Set figure size for better visibility

    # Iterate over all score files in the results directory
    for score_file in glob.glob("results/*_scores.npy"):
        scores = np.load(score_file)  # Load scores from .npy file
        label = os.path.splitext(os.path.basename(score_file))[0]  # Use filename as label
        plt.plot(scores, label=label)  # Plot scores with label

    # Set plot labels and title
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("DDPG Hyperparameter Sweep Comparison")
    plt.legend()         # Show legend for each curve
    plt.grid(True)       # Add grid for readability
    plt.tight_layout()   # Adjust layout to prevent overlap
    plt.show()           # Display the plot

if __name__ == "__main__":
    main()
