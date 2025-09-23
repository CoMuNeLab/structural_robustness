"""
Unit tests for centrality.py
Run with: `pytest tests/test_centrality.py`
"""

import networkx as nx
import numpy as np

from structural_robustness.centrality import compute_centralities, compute_centrality_correlations


def test_compute_centralities_output():
    G = nx.path_graph(6)
    modes = ["random", "degree", "betweenness", "entanglement_small"]

    names, times, dicts = compute_centralities(G, modes=modes)

    assert isinstance(names, list)
    assert isinstance(times, list)
    assert isinstance(dicts, dict)

    assert set(names) == set(modes)
    for name in names:
        assert name in dicts
        assert isinstance(dicts[name], dict)
        assert len(dicts[name]) == len(G)


def test_centrality_correlation_shapes():
    G = nx.erdos_renyi_graph(10, 0.5, seed=1)
    modes = ["random", "degree", "betweenness"]

    _, _, dicts = compute_centralities(G, modes=modes)
    pearson, spearman, mi = compute_centrality_correlations(dicts)

    n = len(modes)
    assert pearson.shape == (n, n)
    assert spearman.shape == (n, n)
    assert mi.shape == (n, n)

    for mat in [pearson, spearman, mi]:
        assert np.allclose(mat, mat.T, atol=1e-8)  # symmetric
        assert np.all(np.isfinite(mat))