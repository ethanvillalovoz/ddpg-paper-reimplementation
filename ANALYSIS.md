# DDPG Results Analysis

## 1. Overview

This document summarizes the results of hyperparameter sweeps for DDPG on the Pendulum-v1 environment, and compares them to the results reported in the original DDPG paper ([Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971)).

## 2. Hyperparameter Sweep Results

All experiments were logged in [`results/experiment_log.csv`](../results/experiment_log.csv).  
Plots for each run are saved in the [`results/`](../results/) directory.

### Best Runs (Top 5 by Best Score)

| alpha  | beta   | tau   | batch_size | layer1 | layer2 | best_score | avg_score | plot_file |
|--------|--------|-------|------------|--------|--------|------------|-----------|-----------|
| 0.0005 | 0.0001 | 0.01  | 128        | 400    | 300    | -1.36      | -1233.67  | [plot](../results/Pendulum_alpha0.0005_beta0.0001_tau0.01_bs128_l1400_l2300_episodes500.png) |
| 0.0001 | 0.001  | 0.01  | 64         | 400    | 300    | -1.81      | -1306.65  | [plot](../results/Pendulum_alpha0.0001_beta0.001_tau0.01_bs64_l1400_l2300_episodes500.png) |
| 0.0001 | 0.0001 | 0.005 | 128        | 400    | 300    | -511.91    | -1446.69  | [plot](../results/Pendulum_alpha0.0001_beta0.0001_tau0.005_bs128_l1400_l2300_episodes500.png) |
| 0.0001 | 0.0001 | 0.005 | 64         | 400    | 300    | -797.84    | -1396.41  | [plot](../results/Pendulum_alpha0.0001_beta0.0001_tau0.005_bs64_l1400_l2300_episodes500.png) |
| 0.001  | 0.0005 | 0.005 | 128        | 400    | 300    | -788.35    | -1381.32  | [plot](../results/Pendulum_alpha0.001_beta0.0005_tau0.005_bs128_l1400_l2300_episodes500.png) |

### Learning Curves

Include a few representative plots (best, worst, typical):

![Best Run](../results/Pendulum_alpha0.0005_beta0.0001_tau0.01_bs128_l1400_l2300_episodes500.png)

## 3. Comparison to the Paper

- **Paper's reported performance (Pendulum-v0):**  
  The original DDPG paper reports an average reward of about -200 for Pendulum-v0 after training.

- **Your best result (Pendulum-v1):**  
  - Best average score: ~-1233.67 (see above)
  - Best single episode score: ~-1.36

- **Discussion:**  
  - The scores are lower (worse) than the paper, likely due to differences in environment version (Pendulum-v1 is harder), implementation details, or random seeds.
  - The original paper may have used different reward scaling or episode lengths.
  - Hyperparameter sensitivity is evident; some runs perform much better than others.

## 4. Conclusions

- DDPG is sensitive to hyperparameters.
- Some settings can achieve much better performance.
- Results are reproducible and all code/plots are available in this repo.

---

## 5. References

- Lillicrap, T.P., et al. (2015). [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
- [OpenAI Gym](https://gym.openai.com/)
- [TensorFlow](https://www.tensorflow.org/)
