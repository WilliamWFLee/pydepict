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

from pydepict.errors import DepicterError

from .consts import (
    ATOM_PATTERNS,
    SAMPLE_SIZE,
    AtomConstraintsCandidates,
    AtomPattern,
    NeighborConstraints,
    NeighborSpec,
)
from .models import Vector
from .utils import none_iter, prune_hydrogens, prune_terminals, set_depict_coords

__all__ = ["depict"]


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


def _match_atom_pattern(
    pattern: AtomPattern, neighbor_spec_to_count: Dict[NeighborSpec, int]
) -> bool:
    """
    Returns if an atom pattern matches a neighbor spec count.

    They match if the set of keys are equal, and the number of the vectors
    for each neighbor spec matches the specified count.
    """
    return pattern.keys() == neighbor_spec_to_count.keys() and all(
        len(pattern[key]) == neighbor_spec_to_count[key] for key in pattern
    )


def _find_candidate_atom_constraints(
    atom_index: int,
    graph: nx.Graph,
) -> List[Tuple[NeighborConstraints, float]]:
    """
    Retrieves all possible candidate atom constraints
    for the atom with the specified index in the specified graph.
    """
    # Determines element of atom
    element = graph.nodes(data="element")[atom_index]
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
                        for vector, neighbor_idx in zip(vectors, neighbor_idxs_perm):
                            new_candidate_pattern[neighbor_idx] = vector
                        if new_candidate_pattern not in new_candidate_patterns:
                            new_candidate_patterns.append(new_candidate_pattern)
            # Iterate over candidate patterns
            for new_candidate_pattern in new_candidate_patterns:
                candidate = (new_candidate_pattern, weight)
                if candidate not in candidates:
                    candidates.append(candidate)

    return candidates


def _add_atom_constraints_to_sample(
    sample: _Constraints,
    atoms: List[int],
    constraints_candidates: AtomConstraintsCandidates,
) -> None:
    """
    Adds non-conflicting atom constraints to depiction sample
    """
    random.shuffle(atoms)
    candidates_copy = {
        atom_index: (neighbor_constraints.copy(), weights.copy())
        for atom_index, (
            neighbor_constraints,
            weights,
        ) in constraints_candidates.items()
    }
    for u in atoms:
        # Sample constraint for current atom
        patterns, weights = candidates_copy[u]
        # Earlier constraint decision could not be reconciled
        if not patterns or not weights:
            continue
        (pattern,) = random.choices(patterns, weights)
        # Delete atom constraints for current atom from constraints copy
        del candidates_copy[u]

        for v, vector in pattern.items():
            # Set vector in graph-wide constraints
            sample[u, v] = vector
            # Remove conflicting constraints for other atoms
            if v in candidates_copy:
                filtered_constraints = []
                filtered_weights = []
                for i, neighbor_constraints in enumerate(candidates_copy[v][0].copy()):
                    if u in neighbor_constraints and neighbor_constraints[u] == -vector:
                        filtered_constraints.append(candidates_copy[v][0][i])
                        filtered_weights.append(candidates_copy[v][1][i])
                candidates_copy[v] = (filtered_constraints, filtered_weights)


def _calculate_depiction_coordinates(
    graph: nx.Graph, constraints: _Constraints
) -> Dict[int, Vector]:
    coordinates = {0: Vector(0, 0)}
    for u, v in nx.dfs_edges(graph, source=0):
        coordinates[v] = coordinates[u] + constraints[u, v]

    return coordinates


def _choose_best_sample(
    coordinates_samples: List[Dict[int, Vector]]
) -> Dict[int, Vector]:
    """
    Selects the best sample, and applies to the graph
    """
    # TODO: Actual best sample implementation
    return coordinates_samples[0]


def depict(graph: nx.Graph) -> None:
    """
    Determines depiction coordinates for the graph.

    Adds to atom attributes to the graph *in-place*,
    thus changing the original input graph to include depiction coordinates.

    :param graph: The graph to calculate depiction coordinates for.
    :type graph: nx.Graph
    """
    # Makes copy of original molecular graph,
    # and removes hydrogens and terminal atoms to produce a "pruned" graph.
    pruned_graph = graph.copy()
    prune_hydrogens(pruned_graph)
    prune_terminals(pruned_graph)

    # Produce a copy list of atom indices
    atoms: List[int] = list(pruned_graph.nodes)
    # Determine and sample constraints
    atom_constraints: Dict[int, List[Tuple[NeighborConstraints, float]]] = {}
    for atom_index in atoms:
        constraints_list = _find_candidate_atom_constraints(atom_index, graph)
        if not constraints_list:
            raise DepicterError(
                f"No candidate constraints found for atom with index {atom_index}"
            )
        atom_constraints[atom_index] = constraints_list

    # Construct final representation of atom constraints
    atom_constraints_candidates: AtomConstraintsCandidates = {
        atom_index: (
            [neighbor_constraints for neighbor_constraints, _ in patterns],
            [weight for _, weight in patterns],
        )
        for atom_index, patterns in atom_constraints.items()
    }

    # Produce constraint samples
    samples = [_Constraints() for _ in range(SAMPLE_SIZE)]
    for sample in samples:
        _add_atom_constraints_to_sample(sample, atoms, atom_constraints_candidates)
    # Convert constraints to Cartesian coordinates
    coordinates_samples = [
        _calculate_depiction_coordinates(graph, sample) for sample in samples
    ]
    best_sample = _choose_best_sample(coordinates_samples)
    for atom_index, coords in best_sample.items():
        set_depict_coords(atom_index, graph, coords)
