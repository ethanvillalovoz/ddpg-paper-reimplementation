"""
main.py

Entry point for training the DDPG agent on the Pendulum-v1 environment.
Supports random hyperparameter search and logs results to CSV.
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import csv
import random
from src.ddpg_agent import DDPGAgent

np.bool8 = bool

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
    print("Available devices:", tf.config.list_physical_devices())
    os.makedirs("results", exist_ok=True)

    # Hyperparameter search space
    NUM_RUNS = 20  # Number of random experiments
    ALPHAS = [0.001, 0.0005, 0.0001]
    BETAS = [0.001, 0.0005, 0.0001]
    TAUS = [0.005, 0.01, 0.02]
    BATCH_SIZES = [64, 128]
    LAYER1_SIZES = [400]
    LAYER2_SIZES = [300]
    N_ACTIONS = 1
    INPUT_DIMS = [3]
    EPISODES = 500
    ENV_NAME = "Pendulum-v1"
    SEED = 0

    # Prepare CSV logging
    csv_path = "results/experiment_log.csv"
    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow([
            'alpha', 'beta', 'tau', 'batch_size', 'layer1_size', 'layer2_size',
            'best_score', 'avg_score', 'plot_file'
        ])

    for run in range(NUM_RUNS):
        ALPHA = random.choice(ALPHAS)
        BETA = random.choice(BETAS)
        TAU = random.choice(TAUS)
        BATCH_SIZE = random.choice(BATCH_SIZES)
        LAYER1_SIZE = random.choice(LAYER1_SIZES)
        LAYER2_SIZE = random.choice(LAYER2_SIZES)

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
            obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), None)
            while not done:
                action = agent.choose_action(obs)
                step_result = env.step(action)
                if len(step_result) == 5:
                    new_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    new_state, reward, done, info = step_result
                agent.remember(obs, action, reward, new_state, done)
                agent.learn()
                obs = new_state
                score += reward
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print(f"[Run {run+1}/{NUM_RUNS}] [alpha={ALPHA}, beta={BETA}, tau={TAU}, bs={BATCH_SIZE}] "
                  f"Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}")

        filename = (
            f"results/Pendulum_"
            f"alpha{ALPHA}_beta{BETA}_tau{TAU}_bs{BATCH_SIZE}_l1{LAYER1_SIZE}_l2{LAYER2_SIZE}_episodes{EPISODES}.png"
        )
        plot_learning(scores, filename=filename, window_size=100)

        best_score = np.max(scores)
        avg_score = np.mean(scores[-100:])
        csv_writer.writerow([
            ALPHA, BETA, TAU, BATCH_SIZE, LAYER1_SIZE, LAYER2_SIZE,
            best_score, avg_score, filename
        ])
        csv_file.flush()

    csv_file.close()