#!/usr/bin/env python3

"""
pydepict.utils

Utility functions

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

from math import sqrt
from typing import Tuple

import networkx as nx


def bond_order_sum(atom_index: int, graph: nx.Graph) -> int:
    """
    Calculates the sum of the orders of the bonds adjacent to the atom
    at the specified index within the specified graph.

    :param atom_index: The index of the atom to find the bond order sum for
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type graph: nx.Graph
    :return: The sum of the orders of the bonds adjacent to the specified atom
    :type: int
    """
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


def depicted_distance(u: int, v: int, graph: nx.Graph) -> float:
    """
    Calculates the depicted distance between two atoms in a molecule graph.

    :param u: The index of one atom
    :type u: int
    :param v: The index of the other atom
    :type v: int
    :param graph: The graph to look in to find the atom data
    :type graph: nx.Graph
    :return: The distance between the two atoms
    :rtype: float
    """
    ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
    vx, vy = graph.nodes[v]["x"], graph.nodes[v]["y"]
    return sqrt((vx - ux) ** 2 + (vy - uy) ** 2)


def average_depicted_bond_length(graph: nx.Graph) -> float:
    """
    Calculates the average length of the representation of bonds
    in a graph that has depiction coordinates.

    Average length is calculated by calculating a mean of the Euclidean distance
    between the two endpoints of each edge.

    :param graph: The graph to calculate the average bond length for
    :type graph: nx.Graph
    :return: The average bond length in the graph, which is 0 if the graph has no edges
    :rtype: float
    """
    if not graph.edges:
        return 0
    total_distance = sum(depicted_distance(u, v, graph) for u, v in graph.edges)
    return total_distance / len(graph.edges)


def get_depict_coords(atom_index: int, graph: nx.Graph) -> Tuple[float, float]:
    """
    Gets depiction coordinates for the atom with the specified index
    in the specified graph.

    :param atom_index: The index of the atom to fetch coordinates for.
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type atom_index: int
    :return: The depiction coordinates for the specified atom
    :rtype: Tuple[float, float]
    """

    x = graph.nodes[atom_index]["x"]
    y = graph.nodes[atom_index]["y"]

    return x, y


def get_display_coords(atom_index: int, graph: nx.Graph) -> Tuple[float, float]:
    """
    Gets display coordinates for the atom with the specified index
    in the specified graph.

    :param atom_index: The index of the atom to fetch coordinates for.
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type atom_index: int
    :return: The display coordinates for the specified atom
    :rtype: Tuple[float, float]
    """

    x = graph.nodes[atom_index]["dx"]
    y = graph.nodes[atom_index]["dy"]

    return x, y


def set_display_coords(
    atom_index: int, graph: nx.Graph, coords: Tuple[float, float]
) -> None:
    """
    Sets display coordinates for the atom with the specified index
    in the specified graph.

    :param atom_index: The index of the atom to set coordinates for.
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type atom_index: int
    :param coords: The display coordinates to set for the specified atom
    :type coords: Tuple[float, float]
    """

    x, y = coords
    graph.nodes[atom_index]["dx"] = x
    graph.nodes[atom_index]["dy"] = y
