"""
Module: entropy.py
==================
Entropy-based metrics for graph robustness analysis using the Laplacian spectrum.

This module includes functions to compute entropy and entanglement measures derived from
spectral graph theory. These metrics can be used to assess the information-theoretic
structural robustness of networks.
"""

import numpy as np
import networkx as nx
import scipy.sparse.linalg

from typing import Tuple

def compute_entropy(G: nx.Graph, beta: float) -> float:
    """
    Compute the von Neumann entropy of a graph using its Laplacian spectrum.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    beta : float
        Diffusion parameter.

    Returns
    -------
    S : float
        The entropy value.
    """
    Ls = np.sort(nx.laplacian_spectrum(G))
    Z = np.sum(np.exp(-beta * Ls))
    p = np.exp(-beta * Ls) / Z
    p = np.delete(p, np.where(p < 1e-20))
    S = np.sum(-p * np.log2(p))
    return S

def compute_entropy_approx(G: nx.Graph, beta: float, k: int = 2) -> float:
    """
    Approximate the entropy of a graph using a truncated Laplacian spectrum.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    beta : float
        Diffusion parameter.
    k : int, optional
        Number of smallest non-zero eigenvalues to compute (default is 2).

    Returns
    -------
    S : float
        The approximated entropy value.
    """
    Ls = scipy.sparse.linalg.eigsh(
        nx.laplacian_matrix(G, weight='weight').astype(float),
        k=k,
        which='SM',
        return_eigenvectors=False
    )
    Z = np.sum(np.exp(-beta * Ls))
    p = np.exp(-beta * Ls) / Z
    p = np.delete(p, np.where(p < 1e-20))
    S = np.sum(-p * np.log2(p))
    return S

def estimate_entropy_and_beta(G: nx.Graph, decay: float = 0.33) -> Tuple[float, float]:
    """
    Estimate the entropy and diffusion parameter beta from the graph's smallest
    non-zero Laplacian eigenvalue.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    decay : float
        Desired probability decay used to estimate beta (default is 0.33).

    Returns
    -------
    S : float
        Entropy of the graph.
    beta : float
        Estimated diffusion parameter.
    """
    Ls = np.sort(nx.laplacian_spectrum(G))
    diff_time = Ls[np.where(Ls > 1e-12)[0]][0]
    beta = -np.log(decay) / diff_time
    S = compute_entropy(G, beta)
    return S, beta