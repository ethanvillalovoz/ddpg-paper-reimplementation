"""
Generate a markdown table of the best DDPG runs from experiment_log.csv.

Usage:
    python scripts/generate_best_runs_table.py
Copy the output and paste it into ANALYSIS.md.
"""

import pandas as pd

def generate_best_runs_markdown(csv_path, n=5):
    df = pd.read_csv(csv_path)
    # Sort by best_score (ascending, since lower is better for Pendulum)
    top = df.sort_values('best_score').head(n)
    cols = ['alpha', 'beta', 'tau', 'batch_size', 'layer1', 'layer2', 'best_score', 'avg_score', 'plot_file']
    md = '| ' + ' | '.join(cols) + ' |\n'
    md += '|--------|--------|-------|------------|--------|--------|------------|-----------|-----------|\n'
    for _, row in top.iterrows():
        plot_link = f"[plot](../results/{row['plot_file'].split('/')[-1]})"
        md += '| ' + ' | '.join(str(row[c]) for c in cols[:-1]) + f' | {plot_link} |\n'
    return md

if __name__ == "__main__":
    csv_path = '../results/experiment_log.csv'
    print(generate_best_runs_markdown(csv_path, n=5))
