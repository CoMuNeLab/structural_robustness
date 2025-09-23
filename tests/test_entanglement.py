"""
Unit tests for entanglement.py
Run with: `pytest tests/test_entanglement.py`
"""

import networkx as nx
import pytest

from structural_robustness.entanglement import entanglement

@pytest.mark.parametrize("mode", ["small", "mid", "large", "approx"])
def test_entanglement_output_structure(mode):
    G = nx.cycle_graph(6)
    ent = entanglement(G, mode=mode)

    assert isinstance(ent, dict)
    assert len(ent) == len(G)

    for v in ent.values():
        assert isinstance(v, float)
        assert not (v is None or v != v)


def test_entanglement_requires_connected_graph():
    G = nx.Graph()
    G.add_nodes_from([0, 1])  # disconnected

    with pytest.raises(IndexError):
        ent = entanglement(G, mode="mid")
