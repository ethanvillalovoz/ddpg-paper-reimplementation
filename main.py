from DDPG import DDPGAgent
import gym
import numpy as np
from utils import plotLearning

if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    agent = DDPGAgent(alpha=0.001, beta=0.001, input_dims=[3], tau=0.005,
                      env=env, batch_size=64, layer1_size=400,
                      layer2_size=300, n_actions=1)
    np.random.seed(0)
    scores = []
    for i in range(1000):
        score = 0
        done = False
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs)
            new_state, reward, done, info = env.step(action)
            agent.remember(obs, action, reward, new_state, done)
            agent.learn()
            obs = new_state
            score += reward
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f"Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}")

    filename = "Pendulum.png"
    plotLearning(scores, filename=filename, window_size=100)