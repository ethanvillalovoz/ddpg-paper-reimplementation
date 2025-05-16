<div align="center">

# 🏆 DDPG: Deep Deterministic Policy Gradient (TensorFlow 2.x)

**Paper Reimplementation & Reproducible Experiments**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ethanvillalovoz/ddpg-paper-reimplementation/blob/main/notebooks/DDPG_Analysis.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ethanvillalovoz/ddpg-paper-reimplementation/main?filepath=notebooks%2FDDPG_Analysis.ipynb)

[Project Repository](https://github.com/ethanvillalovoz/ddpg-paper-reimplementation) • [Original DDPG Paper (Lillicrap et al., 2015)](https://arxiv.org/abs/1509.02971)

---

</div>

<!-- Optional: Project Logo -->
<!-- ![DDPG Logo](assets/logo.png) -->

# ddpg-paper-reimplementation
# Deep Deterministic Policy Gradient (DDPG) in TensorFlow 2.x

This repo implements the Deep Deterministic Policy Gradient (DDPG) algorithm from the paper:

> **Continuous Control with Deep Reinforcement Learning**  
> Lillicrap et al., 2015  
> [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)

---

## Features

- Full DDPG implementation in TensorFlow 2.x and Keras
- Modular code: agent, networks, buffer, noise, and training script
- Compatible with both CPU and GPU (CUDA or Apple Silicon Metal)
- Handles both new and old Gym API
- Learning curve plotting (`Pendulum.png`)
- Easy to extend for other continuous control environments

---

## Environment

- Python 3.9+
- TensorFlow 2.x (tested with tensorflow-macos and tensorflow-metal on Apple Silicon)
- Gym >= 0.26
- matplotlib, numpy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Setup & Usage

### 🟢 **Conda (CPU only, or CUDA GPU on Linux/Windows)**

> **Note:** Apple Silicon GPU acceleration is **not** supported in conda environments.  
> For Apple GPU support, see the next section.

1. **Create and activate a conda environment:**
   ```sh
   conda create -n ddpg python=3.9
   conda activate ddpg
   ```

2. **Install dependencies:**
   ```sh
   conda install tensorflow gym matplotlib numpy
   ```

3. **Run training:**
   ```sh
   python main.py
   ```

4. **Output:**
   - Training progress is printed to the terminal.
   - Learning curve is saved as `Pendulum.png`.

---

### 🍏 **Apple Silicon GPU (M1/M2/M3/M4) — Recommended for Mac**

1. **Create a Python venv (not conda):**
   ```sh
   python3 -m venv tf-macos
   source tf-macos/bin/activate
   ```

2. **Install dependencies:**
   ```sh
   pip install --upgrade pip
   pip install tensorflow-macos tensorflow-metal gym matplotlib numpy
   ```

3. **Run training:**
   ```sh
   python main.py
   ```

4. **Output:**
   - Training progress is printed to the terminal.
   - Learning curve is saved as `Pendulum.png`.

---

## Quick Example

```sh
python main.py
```

---

## Project Status

- ✅ DDPG agent, actor/critic networks, replay buffer, and noise process implemented
- ✅ Modern TensorFlow 2.x code (no sessions/placeholders)
- ✅ Device selection is automatic (CPU/GPU/Apple Silicon)
- ✅ Tested on Pendulum-v1
- ✅ Plotting and logging included

---

## Notes

- For small models and batch sizes, most computation is still on the CPU; GPU is used for neural network operations.
- To see more GPU usage, try larger models or batch sizes.
- The code is ready for further experiments, new environments, or hyperparameter tuning.

---

## Results & Analysis

See [ANALYSIS.md](ANALYSIS.md) for plots, tables, and a comparison with the original DDPG paper.

**Interactive analysis notebook:** [notebooks/DDPG_Analysis.ipynb](notebooks/DDPG_Analysis.ipynb) (try it on Binder or Colab!)

---

## How to cite

If you use this codebase, please cite it using the provided [CITATION.cff](CITATION.cff) file.

---

## License

MIT License

See [LICENSE](LICENSE) for details.

---

## References

- Lillicrap, T.P., et al. (2015). [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
- [OpenAI Gym](https://gym.openai.com/)
- [TensorFlow](https://www.tensorflow.org/)

---
