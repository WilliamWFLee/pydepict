#!/usr/bin/env python3

"""
pydepict.utils

Utility functions.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details.
"""

import datetime as dt
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import networkx as nx

from .consts import HALOGENS
from .models import Vector
from .types import AtomAttribute, BondAttribute, GraphCoordinates, NeighborSpec

__all__ = [
    "bond_order_sum",
    "atom_valence",
    "is_allenal_center",
    "depicted_distance",
    "average_depicted_bond_length",
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


def depicted_distance(u: int, v: int, positions: GraphCoordinates) -> float:
    """
    Calculates the depicted distance between two atoms in a molecule graph.

    :param u: The index of one atom
    :type u: int
    :param v: The index of the other atom
    :type v: int
    :param positions: The dictionary of depiction coordinates
    :type positions: GraphCoordinates
    :return: The distance between the two atoms
    :rtype: float
    """
    coords1, coords2 = positions[u], positions[v]

    return Vector.distance(coords1, coords2)


def average_depicted_bond_length(graph: nx.Graph, positions: GraphCoordinates) -> float:
    """
    Calculates the average length of the representation of bonds
    in a graph that has depiction coordinates.

    Average length is calculated by calculating a mean of the Euclidean distance
    between the two endpoints of each edge.

    :param graph: The graph to calculate the average bond length for
    :type graph: nx.Graph
    :param positions: The dictionary of depiction coordinates
    :type positions: GraphCoordinates
    :return: The average bond length in the graph, which is 0 if the graph has no edges
    :rtype: float
    """
    if not graph.edges:
        return 0
    total_distance = sum(depicted_distance(u, v, positions) for u, v in graph.edges)
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


def element_match(elem1: Optional[str], elem2: Optional[str]) -> bool:
    """
    Determines whether two element symbols match.

    Two element symbols match when the two element symbols are the same,
    or when either or both element symbols is/are :data:`None`.

    Matching also works for special symbols for representing several elements,
    for example ``X`` for halogens will match with ``F``, ``Cl``, etc.

    :param elem1: One of the element symbols.
    :type elem1: str
    :param elem2: The other element symbol.
    :type elem1: str
    :return: Whether the two element symbols match.
    :rtype: bool
    """
    if elem1 == elem2:
        return True
    if any(elem is None for elem in (elem1, elem2)):
        return True
    if sum(elem == "X" for elem in (elem1, elem2)) == 1:
        elem1 = elem2 if elem1 == "X" else elem1
        return elem1 in HALOGENS
    return False


def bond_order_match(order1: Optional[float], order2: Optional[float]) -> bool:
    """
    Determines whether two bond orders match.

    Two bond orders match when the two bond orders are the same,
    or when either or both bond orders is/are :data:`None`.

    :param order1: One of the bond orders.
    :type order1: Optional[float]
    :param order2: The other bond order.
    :type order1: Optional[float]
    :return: Whether the two bond orders match.
    :rtype: bool
    """
    if order1 == order2:
        return True
    if any(order is None for order in (order1, order2)):
        return True
    return False


def neighbor_spec_match(
    neighbor_spec1: NeighborSpec,
    neighbor_spec2: NeighborSpec,
) -> bool:
    """
    Determines whether two neighbor specs match.

    Two neighbor specs match both the elements and bond orders match.

    :param neighbor_spec1: One of the neighbor specs.
    :type neighbor_spec1: NeighborSpec,
    :param neighbor_spec2: The other neighbor specs.
    :type neighbor_spec2: NeighborSpec,
    :return: Whether the two neighbor specs match.
    :rtype: bool
    """
    return all(
        match(spec_value1, spec_value2)
        for match, (spec_value1, spec_value2) in zip(
            (element_match, bond_order_match), zip(neighbor_spec1, neighbor_spec2)
        )
    )


def get_datetime_filename() -> str:
    """
    Returns a human-readable, datetime-based filename without the extension.

    :return: The filename
    :rtype: str
    """
    return dt.datetime.now().strftime("pydepict_%Y_%m_%d_%H_%M_%S")
