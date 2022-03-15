#!/usr/bin/env python3

"""
pydepict.utils

Utility functions

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

import networkx as nx


def bond_order_sum(atom_index: int, graph: nx.Graph) -> int:
    return sum(bond["order"] for bond in graph.adj[atom_index].values())


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

    return (
        bond_order_sum(atom_index, graph)
        + (0 if hcount is None else hcount)
        + atom["charge"]
    )


def is_allenal_center(atom_index: int, graph: nx.Graph) -> bool:
    """
    Determines whether or not the atom at the specified index
    within the specified graph is the carbon at the center of an allene.

    :param atom_index: The index of the atom to calculate valence for
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type graph: nx.Graph
    :return: Whether the atom is an allene center or not
    :type: bool
    """
    atom = graph.nodes[atom_index]
    num_double_bonds = list(
        bond["order"] for bond in graph.adj[atom_index].values()
    ).count(2)

    return (
        atom["element"] == "C"
        and num_double_bonds == 2
        and bond_order_sum(atom_index, graph) == 4
    )
