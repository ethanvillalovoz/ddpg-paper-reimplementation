import matplotlib.pyplot as plt
import numpy as np

def plotLearning(scores, filename, window=100):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    plt.plot(running_avg)
    plt.title('Running average of previous {} scores'.format(window))
    plt.savefig(filename)