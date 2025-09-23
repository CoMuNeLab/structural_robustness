"""
Structural Robustness
=====================
A library for analyzing network structural robustness using entropy, entanglement,
and centrality-driven dismantling strategies.
"""

from .entropy import compute_entropy, compute_entropy_approx, estimate_entropy_and_beta
from .entanglement import entanglement
from .centrality import compute_centralities, compute_centrality_correlations
from .robustness import simulate_dismantling, run_robustness_pipeline
from .utils import random_centrality
from .io import save_numpy_array, load_numpy_array

__all__ = [
    # entropy
    "compute_entropy",
    "compute_entropy_approx",
    "estimate_entropy_and_beta",

    # entanglement
    "entanglement",

    # centrality
    "compute_centralities",
    "compute_centrality_correlations",

    # robustness
    "simulate_dismantling",
    "run_robustness_pipeline",

    # utilities
    "random_centrality",

    # I/O
    "save_numpy_array",
    "load_numpy_array",
]
