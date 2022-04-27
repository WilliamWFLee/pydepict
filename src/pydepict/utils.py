#!/usr/bin/env python3

"""
pydepict.utils

Utility functions

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

import datetime as dt
from itertools import product
from math import sqrt
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import networkx as nx

from .consts import CHAIN_ELEMENTS, AtomAttribute, BondAttribute
from .models import Vector

__all__ = [
    "bond_order_sum",
    "atom_valence",
    "is_allenal_center",
    "depicted_distance",
    "average_depicted_bond_length",
    "get_depict_coords",
    "get_render_coords",
    "set_render_coords",
    "none_iter",
]

T = TypeVar("T")


def get_atom_attrs(
    atom_index: int, graph: nx.Graph, *attrs: str, allow_none: bool = False
) -> Union[AtomAttribute, Tuple[AtomAttribute, ...]]:
    """
    Retrieves atom attributes for the specified atom in the specified graph.

    Accepts one or more attribute names. If one attribute name is provided,
    then that attribute value is returned. If multiple attributes name are provided,
    then a tuple of attribute values is returned, in the order in which
    the attribute names are passed.

    :param atom_index: The index of the atom to look for
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type graph: nx.Graph
    :param *args: One or more attribute names to fetch
    :type *args: str
    :param allow_none: Whether :data:`None` may be returned
                       if an attribute is not found, defaults to :data:`False`
    :type allow_none: bool
    :return: The attribute value, or a tuple of attribute values
    :rtype: Union[AtomAttribute, Tuple[AtomAttribute, ...]]
    """
    atom = graph.nodes[atom_index]
    if len(attrs) == 1:
        return atom.get(attrs[0], None) if allow_none else atom[attrs[0]]
    return (
        tuple(atom.get(attr, None) for attr in attrs)
        if allow_none
        else tuple(atom[attr] for attr in attrs)
    )


def get_bond_attrs(
    u: int, v: int, graph: nx.Graph, *attrs: str, allow_none: bool = True
) -> Union[BondAttribute, Tuple[BondAttribute, ...]]:
    """
    Retrieves bond attributes between the specified atoms in the specified graph.

    Accepts one or more attribute names. If one attribute name is provided,
    then that attribute value is returned. If multiple attributes name are provided,
    then a tuple of attribute values is returned, in the order in which
    the attribute names are passed.

    :param u: One endpoint of the bond
    :type u: int
    :param v: The other endpoint of the bond
    :type v: int
    :param graph: The graph to look for the atom in
    :type graph: nx.Graph
    :param *args: One or more attribute names to fetch
    :type *args: str
    :param allow_none: Whether :data:`None` may be returned
                       if an attribute is not found, defaults to :data:`False`
    :type allow_none: bool
    :return: The attribute value, or a tuple of attribute values
    :rtype: Union[AtomAttribute, Tuple[AtomAttribute, ...]]
    """
    bond = graph[u][v]
    if len(attrs) == 1:
        return bond.get(attrs[0], None) if allow_none else bond[attrs[0]]
    return (
        tuple(bond.get(attr, None) for attr in attrs)
        if allow_none
        else tuple(bond[attr] for attr in attrs)
    )


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


def neighbors(
    atom_index: int, graph: nx.Graph, exclude: Iterable[int] = []
) -> List[int]:
    """
    Determines the neighbors of the specified atom,
    returned as a list of indices.

    :param atom_index: The index of the atom to find neighbors for
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type graph: nx.Graph
    :param exclude: A list of indices to exclude from neighbors
    :type exclude: Iterable[int]
    :return: A list of neighbors
    :rtype: List[int]
    """
    return list(filter(lambda v: v not in exclude, graph[atom_index].keys()))


def num_heavy_atom_neighbors(atom_index: int, graph: nx.Graph) -> int:
    """
    Determines the number of heavy atom neighbors for the specified atom.
    A heavy atom is any atom that is not hydrogen.

    :param atom_index: The index of the atom to find
                       the number of heavy atom neighbors for
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type graph: nx.Graph
    :return: The number of heavy atom neighbors
    :rtype: int
    """
    return sum(
        get_atom_attrs(v, graph, "element") != "H" for v in neighbors(atom_index, graph)
    )


def num_bond_order(atom_index: int, graph: nx.Graph, order: int) -> bool:
    """
    Counts the number of bonds of a particular order.

    :param atom_index: The index of the atom to count for
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type graph: nx.Graph
    :param order: The order of bonds to count
    :type order: int
    :return: The number of bonds of the specified order
    :rtype: int
    """
    return list(bond["order"] == 3 for bond in graph[atom_index].values()).count(order)


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
    return (
        get_atom_attrs(atom_index, graph, "element") == "C"
        and num_bond_order(atom_index, graph, 2) == 2
        and bond_order_sum(atom_index, graph) == 4
    )


def is_chain_atom(atom_index: int, graph: nx.Graph) -> bool:
    """
    Determines whether or not the specified atom in the specified graph
    is a chain atom (i.e. eligible to be treated as being in a chain).

    :param graph: The graph to look for the atom in
    :type graph: nx.Graph
    :param atom_index: The index of the atom
    :type atom_index: int
    :return: Whether the atom is a chain atom
    :rtype: bool
    """
    element, charge = get_atom_attrs(atom_index, graph, "element", "charge")
    return (
        element in CHAIN_ELEMENTS
        and charge == 0
        and num_bond_order(atom_index, graph, 3) == 0
        and num_heavy_atom_neighbors(atom_index, graph) in (2, 3)
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
    ux, uy = get_atom_attrs(u, graph, "dx", "dy")
    vx, vy = get_atom_attrs(v, graph, "dx", "dy")
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


def depiction_width(sample: Dict[int, Vector]) -> float:
    """
    Calculates the depicted width of a depiction sample,
    calculated as the difference between the lowest and highest x coordinates.

    :param sample: The depiction sample
    :type sample: Dict[int, Vector]
    :return: The depicted width
    :rtype: float
    """
    min_x = min(vector.x for vector in sample.values())
    max_x = max(vector.x for vector in sample.values())
    return max_x - min_x


def get_depict_coords(atom_index: int, graph: nx.Graph) -> Vector:
    """
    Gets depiction coordinates for the atom with the specified index
    in the specified graph.

    :param atom_index: The index of the atom to fetch coordinates for.
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type atom_index: int
    :return: The depiction coordinates for the specified atom
    :rtype: Vector
    """
    x, y = get_atom_attrs(atom_index, graph, "dx", "dy")
    return Vector(x, y)


def set_depict_coords(atom_index: int, graph: nx.Graph, coords: Vector) -> None:
    """
    Sets depiction coordinates for the atom with the specified index
    in the specified graph.

    :param atom_index: The index of the atom to set coordinates for.
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type graph: nx.Graph
    :param coords: The depiction coordinates to set for the specified atom
    :type coords: Vector
    """
    graph.nodes[atom_index]["dx"] = coords.x
    graph.nodes[atom_index]["dy"] = coords.y


def get_render_coords(atom_index: int, graph: nx.Graph) -> Vector:
    """
    Gets render coordinates for the atom with the specified index
    in the specified graph.

    :param atom_index: The index of the atom to fetch coordinates for.
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type atom_index: int
    :return: The display coordinates for the specified atom
    :rtype: Vector
    """
    x, y = get_atom_attrs(atom_index, graph, "rx", "ry")

    return Vector(x, y)


def set_render_coords(atom_index: int, graph: nx.Graph, coords: Vector) -> None:
    """
    Sets render coordinates for the atom with the specified index
    in the specified graph.

    :param atom_index: The index of the atom to set coordinates for.
    :type atom_index: int
    :param graph: The graph to look for the atom in
    :type graph: nx.Graph
    :param coords: The render coordinates to set for the specified atom
    :type coords: Vector
    """
    graph.nodes[atom_index]["rx"] = coords.x
    graph.nodes[atom_index]["ry"] = coords.y


def none_iter(iterable: Iterable[T]) -> Iterable[Iterable[Optional[T]]]:
    """
    Takes an iterable, and returns an iterable that iterates all possible
    substitutions of the elements of the iterable with :data:`None`.

    :param iterable: The iterable
    :type iterable: Iterable[T]
    :return: The iterable of iterables with all possible substitutions
    :rtype: Iterable[Iterable[Optional[T]]]
    """
    return product(
        *((value, None) if value is not None else (None,) for value in iterable)
    )


def prune_hydrogens(graph: nx.Graph, atoms: List[int]):
    """
    Finds hydrogen atoms in the graph, and removes them from the specified list
    of atom indices.

    If removing an index would empty the list, then one remaining index is kept.

    :param graph: The graph to find hydrogens within
    :type graph: nx.Graph
    :param atoms: The list to remove atom indices from.
    :type atoms: List[int]
    """
    for atom_index, element in dict(graph.nodes(data="element")).items():
        if element == "H" and atom_index in atoms and len(atoms) > 1:
            atoms.remove(atom_index)


def prune_terminals(graph: nx.Graph, atoms: List[int]):
    """
    Finds terminals atoms in the graph, and removes them from the specified list
    of atom indices.

    If removing an index would empty the list, then one remaining index is kept.

    :param graph: The graph to find terminals within
    :type graph: nx.Graph
    :param atoms: The list to remove atom indices from.
    :type atoms: List[int]
    """
    for atom_index in list(graph.nodes):
        if len(graph[atom_index]) <= 1 and atom_index in atoms and len(atoms) > 1:
            atoms.remove(atom_index)


def get_datetime_filename() -> str:
    """
    Returns a human-readable, datetime-based filename without the extension.

    :return: The filename
    :rtype: str
    """
    return dt.datetime.now().strftime("pydepict_%Y_%m_%d_%H_%M_%S")
