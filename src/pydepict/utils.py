#!/usr/bin/env python3

"""
pydepict.utils

Utility functions
"""

import networkx as nx


def atom_valence(atom_index: int, graph: nx.Graph) -> int:
    """
    Calculates the valence of the atom with the specified index
    within the specified graph.

    :param atom_index: The index of the atom to calculate valence for
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type graph: nx.Graph
    :return: The valence of the atom
    :type: int
    """
    atom = graph.nodes[atom_index]
    hcount = getattr(atom, "hcount", None)
    bond_order_sum = sum(
        bond_attrs["order"] for bond_attrs in graph.adj[atom_index].values()
    )

    return bond_order_sum + (0 if hcount is None else hcount) + atom["charge"]
