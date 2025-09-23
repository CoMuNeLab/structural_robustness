"""
Module: centrality.py
=====================
Provides various centrality measures including standard ones from NetworkX
and custom entanglement-based centrality.
"""

import time
import networkx as nx
from typing import Dict, List, Tuple
import numpy as np
from sklearn import metrics
import scipy.stats as stats

from .utils import random_centrality
from .entanglement import entanglement

def compute_centralities(G: nx.Graph, modes: List[str] = None) -> Tuple[List[str], List[float], Dict[str, Dict]]:
    """
    Compute a collection of centrality measures on the given graph.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    modes : list of str, optional
        List of centrality measures to compute. Defaults to a standard set.

    Returns
    -------
    centrality_names : list of str
        Names of the computed centralities.
    timings : list of float
        Computation times for each centrality.
    centrality_dicts : dict
        Dictionary mapping each centrality name to its node score mapping.
    """
    if modes is None:
        modes = ['random', 'degree', 'betweenness',
                 'entanglement_small', 'entanglement_mid', 'entanglement_large']

    centrality_dicts = {}
    centrality_names = []
    timings = []

    for mode in modes:
        print(f"{mode} centrality analysis started")
        t_start = time.time()

        if mode == 'random':
            scores = random_centrality(G)
        elif mode == 'degree':
            scores = nx.degree_centrality(G)
        elif mode == 'betweenness':
            scores = nx.betweenness_centrality(G, normalized=False, endpoints=True)
        elif mode.startswith('entanglement'):
            ent_mode = mode.split('_')[-1]
            scores = entanglement(G, mode=ent_mode)
        else:
            raise ValueError(f"Unknown centrality mode: {mode}")

        t_end = time.time()
        centrality_names.append(mode)
        timings.append(t_end - t_start)
        centrality_dicts[mode] = scores

    return centrality_names, timings, centrality_dicts

def compute_centrality_correlations(centrality_dicts: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute correlation matrices (Pearson, Spearman, Mutual Info) for a set of centrality values.

    Parameters
    ----------
    centrality_dicts : dict
        Dictionary mapping centrality names to {node: score} dictionaries.

    Returns
    -------
    pearson : np.ndarray
        Pearson correlation matrix.
    spearman : np.ndarray
        Spearman correlation matrix.
    mutual_info : np.ndarray
        Mutual information matrix.
    """
    keys = list(centrality_dicts.keys())
    m = len(keys)
    pearson = np.zeros((m, m))
    spearman = np.zeros((m, m))
    mutual_info = np.zeros((m, m))

    for i in range(m):
        vi = np.array(list(centrality_dicts[keys[i]].values()))
        for j in range(m):
            vj = np.array(list(centrality_dicts[keys[j]].values()))
            pearson[i, j] = stats.pearsonr(vi, vj)[0]
            spearman[i, j] = stats.spearmanr(vi, vj)[0]
            mutual_info[i, j] = metrics.mutual_info_score(vi, vj)

    return pearson, spearman, mutual_info