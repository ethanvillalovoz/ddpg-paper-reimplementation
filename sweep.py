import itertools
import subprocess
import yaml
import shutil
import os
from datetime import datetime

base_config_path = "config.yaml"
with open(base_config_path) as f:
    base_config = yaml.safe_load(f)

taus = [0.001, 0.005, 0.01, 0.02]
batch_sizes = [32, 64, 128, 256]
alphas = [0.00005, 0.0001, 0.0002]
betas = [0.0005, 0.001, 0.002]
layer1_sizes = [200, 400, 800]

os.makedirs("experiments", exist_ok=True)

for tau, batch_size, alpha, beta, layer1_size in itertools.product(
    taus, batch_sizes, alphas, betas, layer1_sizes
):
    config = yaml.safe_load(open(base_config_path))  # fresh copy each time
    config['agent']['tau'] = tau
    config['agent']['batch_size'] = batch_size
    config['agent']['alpha'] = alpha
    config['agent']['beta'] = beta
    config['agent']['layer1_size'] = layer1_size
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = (
        f"experiments/config_tau{tau}_bs{batch_size}_alpha{alpha}_beta{beta}_l1{layer1_size}_{timestamp}.yaml"
    )
    with open(config_name, "w") as f:
        yaml.dump(config, f)
    print(f"Running with tau={tau}, batch_size={batch_size}, alpha={alpha}, beta={beta}, layer1_size={layer1_size}")
    subprocess.run(["python", "src/main.py", "--config", config_name])
