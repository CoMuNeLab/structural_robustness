"""
Module: io.py
=============
Input/output utility functions for saving and loading numerical results.
"""

import numpy as np
from typing import Any

def save_numpy_array(filename: str, array: Any) -> None:
    """
    Save a NumPy array or dictionary to disk.

    Parameters
    ----------
    filename : str
        Path to the output `.npy` file (without extension).
    array : array-like or dict
        Data to be saved.
    """
    np.save(filename, array)


def load_numpy_array(filename: str) -> Any:
    """
    Load a NumPy array or dictionary from disk.

    Parameters
    ----------
    filename : str
        Path to the `.npy` file (with or without extension).

    Returns
    -------
    array : Any
        Loaded data.
    """
    if not filename.endswith('.npy'):
        filename += '.npy'
    return np.load(filename, allow_pickle=True).item()
