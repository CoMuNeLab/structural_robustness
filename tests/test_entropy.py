"""
Unit tests for entropy.py
Run with: `pytest tests/test_entropy.py`
"""

import networkx as nx
import numpy as np
import pytest

from structural_robustness.entropy import compute_entropy, compute_entropy_approx, estimate_entropy_and_beta

def test_compute_entropy():
    G = nx.path_graph(5)
    beta = 0.5
    S = compute_entropy(G, beta)
    assert isinstance(S, float)
    assert S > 0

def test_compute_entropy_approx():
    G = nx.path_graph(5)
    beta = 0.5
    S = compute_entropy_approx(G, beta)
    assert isinstance(S, float)
    assert S > 0

def test_estimate_entropy_and_beta():
    G = nx.path_graph(5)
    S, beta = estimate_entropy_and_beta(G, decay=0.33)
    assert isinstance(S, float)
    assert isinstance(beta, float)
    assert S > 0
    assert beta > 0