# Structural Robustness

A modular Python library for analyzing the structural robustness of complex networks
using spectral entropy, entanglement centrality, and centrality-based dismantling strategies.

---

## Features
- **Entropy-based analysis**: Compute graph entropy via Laplacian spectrum.
- **Entanglement centrality**: Quantify node importance based on entropy shifts.
- **Centrality toolbox**: Combine classical and custom centralities.
- **Dismantling simulation**: Track network collapse under targeted node removal.
- **Correlation metrics**: Compare centrality measures using Pearson, Spearman, and mutual info.

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone ihttps://github.com/CoMuNeLab/structural_robustness.git
cd structural_robustness
pip install -e .
```

---

## Example Usage

```python
import networkx as nx
from structural_robustness import run_robustness_pipeline

# Generate random graph
G = nx.erdos_renyi_graph(n=200, p=0.05, seed=42)

# Run full pipeline
centralities, timings, scores, curves = run_robustness_pipeline(G)
```

For a full working script, see [`examples/example_usage.py`](examples/example_usage.py).

---

## Tests 
for running tests execute:

```
pytest -q
```

---

## Package Structure

```
├── example
│   └── example_usage.py
├── LICENSE
├── pyproject.toml
├── README.md
├── src
│   └── structural_robustness
│       ├── centrality.py
│       ├── entanglement.py
│       ├── entropy.py
│       ├── __init__.py
│       ├── io.py
│       ├── robustness.py
│       └── utils.py
└── tests
    ├── test_centrality.py
    ├── test_entanglement.py
    ├── test_entropy.py
    └── test_robustness.py
```

---

## Modules Overview

| Module          | Purpose                                           |
|----------------|---------------------------------------------------|
| `entropy.py`   | Spectral entropy computation                     |
| `entanglement.py` | Node impact via entropy removal shift        |
| `centrality.py`| Standard + entanglement centralities             |
| `robustness.py`| Simulate LCC collapse under node attacks         |
| `utility.py`   | Miscellaneous tools (random centrality, etc.)    |
| `io.py`        | Load/save NumPy data                             |

---

## Requirements
- Python 3.8+
- `networkx`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `progressbar2`
