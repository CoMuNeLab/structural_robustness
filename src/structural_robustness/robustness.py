"""
Module: robustness.py
======================
Robustness analysis of network structure via node removal strategies.

This module simulates network dismantling processes by removing nodes in order of centrality
and tracking the size of the largest connected component (LCC) over time.
"""

import time
import numpy as np
import networkx as nx
import operator
from typing import Dict, List, Tuple

from .centrality import compute_centralities


def simulate_dismantling(G: nx.Graph, centrality_dicts: Dict[str, Dict]) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Simulate structural robustness under targeted node removals based on centrality.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    centrality_dicts : dict
        Dictionary of centrality scores per node for each strategy.

    Returns
    -------
    centrality_names : list of str
        Names of centralities used for attacks.
    robustness_curves : dict
        Maps each centrality to a list with LCC sizes after each node removal.
    """
    centrality_names = list(centrality_dicts.keys())
    robustness_curves = {}

    print("######## network robustness analysis started ########")
    for name in centrality_names:
        G_copy = G.copy()
        lcc_sizes = []
        scores = centrality_dicts[name].copy()

        for _ in range(G.number_of_nodes()):
            if len(G_copy) > 1:
                lcc = max(nx.connected_components(G_copy), key=len)
                lcc_sizes.append(len(lcc))
                next_node = max(scores.items(), key=operator.itemgetter(1))[0]
                G_copy.remove_node(next_node)
                scores.pop(next_node)
            else:
                lcc_sizes.append(0)

        robustness_curves[name] = lcc_sizes

    print("######## robustness analysis done ########")
    return centrality_names, robustness_curves


def run_robustness_pipeline(G: nx.Graph, modes: List[str] = None) -> Tuple[List[str], List[float], Dict[str, Dict], Dict[str, List[int]]]:
    """
    Complete robustness analysis pipeline: computes centralities, dismantles graph,
    and returns results.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    modes : list of str, optional
        Centrality modes to use. Default includes common and entanglement-based ones.

    Returns
    -------
    centrality_names : list of str
        Names of the computed centralities.
    timings : list of float
        Timing information for each centrality.
    centrality_dicts : dict
        Dictionary of node-level centralities.
    robustness_curves : dict
        LCC size during node removals per strategy.
    """
    centrality_names, timings, centrality_dicts = compute_centralities(G, modes=modes)
    _, robustness_curves = simulate_dismantling(G, centrality_dicts)
    return centrality_names, timings, centrality_dicts, robustness_curves