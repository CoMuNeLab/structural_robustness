"""
Module: utils.py
==================
Helper functions and utilities for the structural robustness analysis library.
"""

import numpy as np
import networkx as nx
from random import seed, random
from typing import Dict


def random_centrality(G: nx.Graph, seed_value: int = 1) -> Dict:
    """
    Assign random centrality scores to each node.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    seed_value : int, optional
        Seed for reproducibility (default is 1).

    Returns
    -------
    rnd : dict
        Dictionary mapping each node to a random value.
    """
    seed(seed_value)
    rnd = {}
    for node in G.nodes():
        rnd[node] = random()
    return rnd
