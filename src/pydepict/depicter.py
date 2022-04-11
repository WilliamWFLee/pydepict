#!/usr/bin/env python3

"""
pydepict.depicter

Depicter for determining the graphical placement of atoms in a molecular graph.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

import random
from collections import defaultdict
from itertools import permutations, product
from typing import Dict, List, Tuple

import networkx as nx

from .consts import (
    ATOM_PATTERNS,
    SAMPLE_SIZE,
    AtomConstraintsCandidates,
    AtomPattern,
    NeighborConstraints,
    NeighborSpec,
)
from .models import Vector
from .utils import none_iter

__all__ = ["Depicter", "depict"]


class _Constraints:
    """
    Implements an endpoint order-independent data structure
    for storing chosen constraints.

    Endpoints are ordered numerically when setting the constraint vector.
    Vectors are returned in the direction that corresponds with the order
    that the endpoints are presented in.
    """

    def __init__(self):
        self._dict: Dict[int, Dict[int, Vector]] = defaultdict(lambda: {})

    @staticmethod
    def _sort_key(key: Tuple[int, int]) -> Tuple[Tuple[int, int], bool]:
        u, v = key
        if u > v:
            return (v, u), True
        return key, False

    def set_atom_constraints(self, constraints: Dict[int, Dict[int, Vector]]):
        for u, neighbor_constraints in constraints.items():
            for v, vector in neighbor_constraints.items():
                self.__setitem__((u, v), vector)

    def __contains__(self, key: Tuple[int, int]) -> bool:
        (u, v), _ = self._sort_key(key)
        return u in self._dict and v in self._dict[u]

    def __getitem__(self, key: Tuple[int, int]) -> Vector:
        (u, v), flipped = self._sort_key(key)
        if self.__contains__((u, v)):
            return -self._dict[u][v] if flipped else self._dict[u][v]
        raise KeyError(key)

    def __setitem__(self, key: Tuple[int, int], value: Vector):
        (u, v), flipped = self._sort_key(key)
        self._dict[u][v] = -value if flipped else value

    def __delitem__(self, key: Tuple[int, int]):
        (u, v), _ = self._sort_key(key)
        if self.__contains__((u, v)):
            del self._dict[u][v]
        raise KeyError(key)


class Depicter:
    """
    Depicter class for depicting molecular graphs.

    A depicter takes a graph representation of a chemical structure,
    and determines the coordinates of each atom in order for the graph
    to be graphically represented.
    """

    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph

    def _prune_hydrogens(self) -> None:
        """
        Remove hydrogens and their adjacent edges from the pruned graph instance.
        """
        for atom_index, element in dict(
            self._pruned_graph.nodes(data="element")
        ).items():
            if element == "H":
                self._pruned_graph.remove_node(atom_index)

    def _prune_terminals(self) -> None:
        """
        Remove terminal atoms and their adjacent edges from the pruned graph instance.
        """
        for atom_index in list(self._pruned_graph.nodes):
            if len(self._pruned_graph[atom_index]) <= 1:
                self._pruned_graph.remove_node(atom_index)

    def _determine_atom_constraints(self) -> None:
        """
        Finds all candidate constraints for atoms, and removes those
        that cannot be fulfilled if applicable
        """
        # Construct immediate representation of atom constraints
        atom_constraints: Dict[int, List[Tuple[NeighborConstraints, float]]] = {}
        for atom_index in self._atoms:
            constraints_list = _find_candidate_atom_constraints(atom_index, self.graph)
            atom_constraints[atom_index] = constraints_list

        # Construct final representation of atom constraints
        self._atom_constraints: Dict[int, AtomConstraintsCandidates] = {
            atom_index: (
                [neighbor_constraints for neighbor_constraints, _ in patterns],
                [weight for _, weight in patterns],
            )
            for atom_index, patterns in atom_constraints.items()
        }

    def _sample_atom_constraints(self):
        """
        Adds non-conflicting atom constraints to depiction samples
        """
        for constraints in self._samples:
            random.shuffle(self._atoms)
            atom_constraints_copy = {
                atom_index: (neighbor_constraints.copy(), weights.copy())
                for atom_index, (
                    neighbor_constraints,
                    weights,
                ) in self._atom_constraints.items()
            }
            for u in self._atoms:
                # Sample constraint for current atom
                patterns, weights = atom_constraints_copy[u]
                (pattern,) = random.choices(patterns, weights)
                # Delete atom constraints for current atom from constraints copy
                del atom_constraints_copy[u]

                for v, vector in pattern.items():
                    # Set vector in graph-wide constraints
                    constraints[u, v] = vector
                    # Remove conflicting constraints for other atoms
                    if v in atom_constraints_copy:
                        removed = 0
                        for i, neighbor_constraints in enumerate(
                            atom_constraints_copy[v][0].copy()
                        ):
                            if (
                                u in neighbor_constraints
                                and neighbor_constraints[u] != -vector
                            ):
                                atom_constraints_copy[v][0].pop(i - removed)
                                atom_constraints_copy[v][1].pop(i - removed)
                                removed += 1

    def _sample(self):
        self._samples = [_Constraints() for _ in range(SAMPLE_SIZE)]
        self._sample_atom_constraints()

    def depict(self) -> None:
        """
        Determines depiction coordinates for the graph in this depicter.

        Adds to atom attributes to the graph *in-place*,
        thus changing the original input graph to include depiction coordinates.
        """
        # Makes copy of original molecular graph,
        # and removes hydrogens and terminal atoms to produce a "pruned" graph.
        self._pruned_graph = self.graph.copy()
        self._prune_hydrogens()
        self._prune_terminals()

        # Produce list of atom indices
        self._atoms: List[int] = list(self._pruned_graph.nodes)
        self._determine_atom_constraints()
        self._sample()

        del self._pruned_graph, self._atoms


def _match_atom_pattern(
    pattern: AtomPattern, neighbor_spec_to_count: Dict[NeighborSpec, int]
) -> bool:
    """
    Returns if an atom pattern matches a neighbor spec count.

    They match if the set of keys are equal, and the number of the vectors
    for each neighbor spec matches the specified count.
    """
    return pattern.keys() == neighbor_spec_to_count.keys() and all(
        len(vectors) == count
        for vectors, count in zip(pattern.values(), neighbor_spec_to_count.values())
    )


def _find_candidate_atom_constraints(
    atom_index: int,
    graph: nx.Graph,
) -> List[Tuple[Dict[int, Vector], float]]:
    """
    Retrieves all possible candidate atom constraints
    for the atom with the specified index in the specified graph.
    """
    # Determines element of atom
    atom_element = graph.nodes(data="element")[atom_index]
    # Finds all candidate patterns
    patterns = ATOM_PATTERNS[atom_element]
    # Get neighbor data
    neighbors_idxs = tuple(graph[atom_index])
    neighbor_elements, neighbor_bond_orders = zip(
        *(
            (graph.nodes[u]["element"], graph[u][atom_index]["order"])
            for u in neighbors_idxs
        )
    )
    # Determine candidates
    candidates = []
    # Iterate over possibilities of neighbor being connected via any bond
    # or being any element
    for elements, bond_orders in product(
        none_iter(neighbor_elements), none_iter(neighbor_bond_orders)
    ):
        # Map (element, order) pairs to counts, and to list of indices
        neighbor_spec_to_count = defaultdict(lambda: 0)
        neighbor_spec_to_idxs = defaultdict(lambda: [])
        for neighbor_idx, element, bond_order in zip(
            neighbors_idxs, elements, bond_orders
        ):
            neighbor_spec = (element, bond_order)
            neighbor_spec_to_idxs[neighbor_spec].append(neighbor_idx)
            neighbor_spec_to_count[neighbor_spec] += 1

        for element in (None, atom_element):
            patterns = ATOM_PATTERNS[element]
            for pattern, weight in patterns:
                if not _match_atom_pattern(pattern, neighbor_spec_to_count):
                    continue
                new_candidate_patterns = [{}]
                # Iterate over vectors for each neighbor spec
                for neighbor_spec, vectors in pattern.items():
                    # Copy current partial new candidates
                    prev_candidate_patterns = new_candidate_patterns.copy()
                    new_candidate_patterns.clear()
                    neighbor_idxs = neighbor_spec_to_idxs[neighbor_spec]
                    # Iterate over each partial new candidate
                    for prev_candidate_pattern in prev_candidate_patterns:
                        # Iterate over permutations of vectors
                        for neighbor_idxs_perm in permutations(neighbor_idxs):
                            new_candidate_pattern = prev_candidate_pattern.copy()
                            for vector, neighbor_idx in zip(
                                vectors, neighbor_idxs_perm
                            ):
                                new_candidate_pattern[neighbor_idx] = vector
                            if new_candidate_pattern not in new_candidate_patterns:
                                new_candidate_patterns.append(new_candidate_pattern)
                # Iterate over candidate patterns
                for new_candidate_pattern in new_candidate_patterns:
                    candidate = (new_candidate_pattern, weight)
                    if candidate not in candidates:
                        candidates.append(candidate)

    return candidates


def depict(graph: nx.Graph) -> None:
    """
    Shortcut that uses that :class:`Depicter`
    to add depiction coordinates to the input graph.

    It is equivalent to::
        Depicter(graph).depict()

    :param graph: The graph to depict
    :type graph: nx.Graph
    """
    depicter = Depicter(graph)
    depicter.depict()
