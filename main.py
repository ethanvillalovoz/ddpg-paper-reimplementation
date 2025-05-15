"""
main.py

Entry point for training the DDPG agent on the Pendulum-v1 environment.
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from src.ddpg_agent import DDPGAgent

def plot_learning(scores, filename, window_size=100):
    """
    Plot the running average of scores and save to file.
    """
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window_size):(t+1)])
    plt.plot(running_avg)
    plt.title(f'Running Average of Previous {window_size} Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Hyperparameters
    ENV_NAME = "Pendulum-v1"
    EPISODES = 1000
    ALPHA = 0.001
    BETA = 0.001
    TAU = 0.005
    BATCH_SIZE = 64
    LAYER1_SIZE = 400
    LAYER2_SIZE = 300
    N_ACTIONS = 1
    INPUT_DIMS = [3]
    SEED = 0

    env = gym.make(ENV_NAME)
    np.random.seed(SEED)
    agent = DDPGAgent(
        alpha=ALPHA, beta=BETA, input_dims=INPUT_DIMS, tau=TAU,
        env=env, batch_size=BATCH_SIZE, layer1_size=LAYER1_SIZE,
        layer2_size=LAYER2_SIZE, n_actions=N_ACTIONS
    )

    scores = []
    for i in range(EPISODES):
        score = 0
        done = False
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(obs, action, reward, new_state, done)
            agent.learn()
            obs = new_state
            score += reward
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f"Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}")

    plot_learning(scores, filename="Pendulum.png", window_size=100)