"""
Unit tests for robustness.py
Run with: `pytest tests/test_robustness.py`
"""

import networkx as nx
import numpy as np

from structural_robustness.robustness import run_robustness_pipeline, simulate_dismantling
from structural_robustness.centrality import compute_centralities


def test_run_robustness_pipeline_shapes():
    G = nx.erdos_renyi_graph(20, 0.2, seed=42)
    modes = ["degree", "entanglement_small"]

    names, times, scores, curves = run_robustness_pipeline(G, modes=modes)

    assert isinstance(names, list)
    assert isinstance(times, list)
    assert isinstance(scores, dict)
    assert isinstance(curves, dict)

    for name in names:
        assert name in curves
        assert len(curves[name]) == G.number_of_nodes()
        assert all(isinstance(v, int) for v in curves[name])


def test_simulate_dismantling_lcc_decreases():
    G = nx.barabasi_albert_graph(30, 2, seed=0)
    modes = ["degree"]

    _, _, scores = compute_centralities(G, modes)
    _, curves = simulate_dismantling(G, scores)

    lcc_sizes = curves["degree"]
    assert len(lcc_sizes) == G.number_of_nodes()
    assert all(isinstance(x, int) for x in lcc_sizes)

    # Check that LCC size generally decreases
    assert lcc_sizes[0] >= lcc_sizes[-1]