"""
Example: Structural Robustness Pipeline
======================================
This script demonstrates how to run the full structural robustness pipeline
on a synthetic graph using the modular library.
"""

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from structural_robustness import (
    run_robustness_pipeline,
    compute_centrality_correlations,
    save_numpy_array,
)

# Generate a synthetic graph (Erdos-Renyi)
G = nx.erdos_renyi_graph(n=5000, p=0.05, seed=42)

# Run robustness analysis using selected centrality strategies
modes = ["random", "degree", "betweenness", "entanglement_small", "entanglement_mid", "entanglement_large"]
cen_names, timings, cen_dicts, robustness = run_robustness_pipeline(G, modes)

# Save outputs
save_numpy_array("centrality_scores", cen_dicts)
save_numpy_array("robustness_curves", robustness)

# Compute centrality correlations
pearson, spearman, mutual_info = compute_centrality_correlations(cen_dicts)

# Plot LCC dismantling curves
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

for name in cen_names:
    plt.plot(robustness[name], label=name)

plt.xlabel("Nodes removed")
plt.ylabel("Size of Largest Connected Component")
plt.title("Network Robustness Under Node Removal")
plt.legend()
plt.tight_layout()
plt.show()
