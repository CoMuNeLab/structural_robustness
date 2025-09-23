"""
Module: entanglement.py
=======================
Node entanglement measures based on entropy variations caused by node removal.

This module defines functions to compute the entanglement of each node in a graph,
quantifying its structural importance based on how much its removal affects
the overall spectral entropy.
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple
from progressbar import ProgressBar

from .entropy import compute_entropy, compute_entropy_approx, estimate_entropy_and_beta

def entanglement(G: nx.Graph, mode: str = 'mid') -> Dict:
    """
    Compute node entanglement values based on entropy changes from node removal.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    mode : str, optional
        Controls beta estimation: one of {'small', 'mid', 'large', 'approx'}.

    Returns
    -------
    ent : dict
        Dictionary mapping node to its entanglement value.
    """
    pbar = ProgressBar()

    if mode == 'approx':
        S_1, beta = estimate_entropy_and_beta(G, decay=0.33)
        entropy_func = compute_entropy_approx
    elif mode == 'small':
        S_1, beta = estimate_entropy_and_beta(G, decay=0.90)
        entropy_func = compute_entropy
    elif mode == 'large':
        S_1, beta = estimate_entropy_and_beta(G, decay=0.01)
        entropy_func = compute_entropy
    elif mode =='mid':  # default to 'mid'
        S_1, beta = estimate_entropy_and_beta(G, decay=0.33)
        entropy_func = compute_entropy
    else:
        raise ValueError("Invalid mode. Choose from {'small', 'mid', 'large', 'approx'}.")

    ent = {}
    for node in pbar(G.nodes()):
        G_i = G.copy()
        k = G_i.degree[node]
        G_i.remove_node(node)

        S_2 = entropy_func(G_i, beta)
        G_star = nx.star_graph(k + 1)
        S_star = compute_entropy(G_star, beta)

        S_2 += S_star
        ent[node] = S_2 - S_1

    return ent