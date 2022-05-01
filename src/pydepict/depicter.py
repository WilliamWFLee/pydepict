#!/usr/bin/env python3

"""
pydepict.depicter

Depicter for determining the graphical placement of atoms in a molecular graph.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

import random
from collections import defaultdict
from copy import deepcopy
from itertools import combinations, cycle, permutations, product
from typing import Dict, Generator, List, Tuple

import networkx as nx

from .consts import (
    ATOM_PATTERNS,
    CHAIN_PATTERN_UNITS,
    EPSILON,
    SAMPLE_SIZE,
    THIRTY_DEGREES,
    AtomPattern,
    ConstraintsCandidates,
    NeighborConstraints,
    NeighborSpec,
)
from .errors import DepicterError
from .models import DepictionConstraints, Matrix, Vector
from .utils import (
    depiction_width,
    is_chain_atom,
    neighbors,
    none_iter,
    prune_hydrogens,
    prune_terminals,
    set_depict_coords,
)

__all__ = ["depict"]


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


def _find_atom_constraints(
    atom_index: int,
    graph: nx.Graph,
) -> List[Tuple[NeighborConstraints, float]]:
    """
    Retrieves all possible atom constraints
    for the atom with the specified index in the specified graph.
    """
    # Determines element of atom
    element = graph.nodes(data="element")[atom_index]
    # Get neighbor data
    neighbors_idxs = tuple(graph[atom_index])
    if not neighbors_idxs:
        # Atom has no neighbors
        return [({}, 1)]
    neighbor_elements, neighbor_bond_orders = zip(
        *(
            (graph.nodes[u]["element"], graph[u][atom_index]["order"])
            for u in neighbors_idxs
        )
    )
    # TODO: "X" for halogens
    patterns = ATOM_PATTERNS[element if element in ATOM_PATTERNS else None]
    # Determine candidates
    candidates = []
    # Iterate over possibilities of neighbor being connected via any bond
    # or being any element
    for neighbor_elements, neighbor_bond_orders in product(
        none_iter(neighbor_elements), none_iter(neighbor_bond_orders)
    ):
        # Map (element, order) pairs to counts, and to list of indices
        neighbor_spec_to_count = defaultdict(lambda: 0)
        neighbor_spec_to_idxs = defaultdict(lambda: [])
        for neighbor_idx, neighbor_element, neighbor_bond_order in zip(
            neighbors_idxs, neighbor_elements, neighbor_bond_orders
        ):
            neighbor_spec = (neighbor_element, neighbor_bond_order)
            neighbor_spec_to_idxs[neighbor_spec].append(neighbor_idx)
            neighbor_spec_to_count[neighbor_spec] += 1
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


def _find_chains(atoms: List[int], graph: nx.Graph) -> List[List[int]]:
    """
    Finds all chains in a graph
    """
    unchained_chain_atoms = [
        atom_index for atom_index in atoms if is_chain_atom(atom_index, graph)
    ]
    chains = []
    while len(unchained_chain_atoms) >= 4:
        possible_chains = []
        for u, v in permutations(unchained_chain_atoms, r=2):
            if not nx.has_path(graph, u, v):
                continue
            possible_chains.extend(nx.all_simple_paths(graph, u, v))
        if not possible_chains:
            # No chains possible
            break
        possible_chains = list(
            filter(
                lambda chain: all(u in unchained_chain_atoms for u in chain),
                possible_chains,
            )
        )
        if not possible_chains:
            # No chains that contains atoms that are not in chains already
            break
        # Select the longest chain
        longest_chain = max(*possible_chains, key=lambda chain: len(chain))
        if len(longest_chain) < 4:
            break
        chains.append(longest_chain)
        for atom_index in longest_chain:
            unchained_chain_atoms.remove(atom_index)

    return chains


def _chain_triplets(
    chain: List[int], graph: nx.Graph
) -> Generator[Tuple[int, int, int, Tuple[int, ...]], None, None]:
    left = None
    root_neighbors = neighbors(chain[0], graph)
    other_chain_atoms = [v for v in root_neighbors if v != chain[1]]
    if other_chain_atoms:
        left = other_chain_atoms[0]
    subs = [v for v in root_neighbors if v not in (left, chain[1])]
    yield left, chain[0], chain[1], subs

    for left, root, right in zip(chain[:-2], chain[1:-1], chain[2:]):
        subs = neighbors(root, graph, (left, right))
        yield left, root, right, subs

    right = None
    root_neighbors = neighbors(chain[-1], graph)
    other_chain_atoms = [v for v in root_neighbors if v != chain[-2]]
    if other_chain_atoms:
        right = other_chain_atoms[0]
    subs = [v for v in root_neighbors if v not in (chain[-2], right)]
    yield chain[-2], chain[-1], right, subs


def _find_chain_constraints(atoms: List[int], graph: nx.Graph) -> ConstraintsCandidates:
    """
    Returns a set of constraints for chains
    """
    chains = _find_chains(atoms, graph)
    candidates: ConstraintsCandidates = {}
    for chain in chains:
        patterns = []
        for pattern_units in CHAIN_PATTERN_UNITS:
            pattern = []
            for (left, _, right, subs), chain_pattern in zip(
                _chain_triplets(chain, graph), cycle(pattern_units)
            ):
                neighbor_constraints = {}
                (
                    neighbor_constraints[left],
                    neighbor_constraints[right],
                ) = chain_pattern[0]
                if subs:
                    sub_vectors = chain_pattern[1][len(subs)]
                    for sub, vector in zip(subs, sub_vectors):
                        neighbor_constraints[sub] = vector
                pattern.append(neighbor_constraints)
            patterns.append(pattern)
        weights = [1 for _ in range(len(CHAIN_PATTERN_UNITS))]
        candidates[tuple(chain)] = (patterns, weights)
        for atom_index in chain:
            atoms.remove(atom_index)

    return candidates


def _remove_conflicting_constraints(
    u: int,
    v: int,
    vector: Vector,
    constraints_candidates: ConstraintsCandidates,
) -> None:
    """
    Removes constraints candidate conflicts.

    Returns whether there are constraints left for all blocks
    """
    for block, (patterns, weights) in constraints_candidates.items():
        if v not in block:
            continue
        index = block.index(v)
        filtered_patterns = []
        filtered_weights = []
        for pattern, weight in zip(patterns, weights):
            neighbor_constraints = pattern[index]
            if u in neighbor_constraints and neighbor_constraints[u] == -vector:
                filtered_patterns.append(pattern)
                filtered_weights.append(weight)
        if not filtered_patterns:
            return False
        constraints_candidates[block] = (filtered_patterns, filtered_weights)

    return True


def _sample_constraints(
    sample: DepictionConstraints,
    constraints_candidates: ConstraintsCandidates,
) -> None:
    """
    Adds non-conflicting constraints to depiction sample
    """
    candidates_copy = deepcopy(constraints_candidates)
    # Shuffle order in which fragments are considered
    blocks = list(constraints_candidates.keys())
    random.shuffle(blocks)
    for block in blocks:
        # Sample constraint for current block
        patterns, weights = candidates_copy[block]
        # Earlier constraint decision does not work with other constraints
        if not patterns or not weights:
            return False
        ((pattern, weight),) = random.choices(list(zip(patterns, weights)), weights)
        # Delete constraints for current block from constraints
        del candidates_copy[block]

        for u, neighbor_constraints in zip(block, pattern):
            sample.weights[u] = weight
            for v, vector in neighbor_constraints.items():
                # Set vector in graph-wide constraints sample
                sample[u, v] = vector
                # Remove conflicting constraints
                constraints_left = _remove_conflicting_constraints(
                    u, v, vector, candidates_copy
                )
                if not constraints_left:
                    return False
    return True


def _apply_depiction_sample(
    graph: nx.Graph, constraints: DepictionConstraints
) -> Dict[int, Vector]:
    """
    Applies constraints to a graph to produce a dictionary mapping atom index
    to the position vector of that atom.
    """
    coordinates = {0: Vector(0, 0)}
    for u, v in nx.dfs_edges(graph, source=0):
        coordinates[v] = coordinates[u] + constraints[u, v]

    return coordinates


def _congestion(sample: Dict[int, Vector], weights: Dict[int, float], graph: nx.Graph):
    """
    Calculates the congestion of the sample.
    """
    congestion = 0
    for component in nx.connected_components(graph):
        for u, v in combinations(component, 2):
            if not graph.has_edge(u, v):
                congestion += 1 / (
                    (Vector.distance(sample[u], sample[v]) + EPSILON) ** 2
                    * (weights[u] if u in weights else 1)
                    * (weights[v] if v in weights else 1)
                )

    return congestion


def _choose_best_sample(
    coordinates_samples_with_weights: List[Tuple[Dict[int, Vector], Dict[int, float]]],
    graph: nx.Graph,
) -> Dict[int, Vector]:
    """
    Selects the best sample from a list of dictionaries of coordinate samples,
    """
    best_sample, _ = min(
        coordinates_samples_with_weights,
        key=lambda sample_with_weight: _congestion(
            sample_with_weight[0], sample_with_weight[1], graph
        ),
    )
    return best_sample


def _maximize_sample_width(sample: Dict[int, Vector]):
    """
    Rotates a depiction sample such that its width is maximized.
    """
    matrices = [Matrix.rotate(THIRTY_DEGREES * i) for i in range(12)]
    widest_sample = max(
        (
            {atom_index: matrix * vector for atom_index, vector in sample.items()}
            for matrix in matrices
        ),
        key=depiction_width,
    )
    sample.update(widest_sample)


def _postprocess_sample(sample: Dict[int, Vector]):
    """
    Postprocesses a sample dictionary to produce the final depiction
    """
    _maximize_sample_width(sample)


def depict(graph: nx.Graph) -> None:
    """
    Determines depiction coordinates for the graph.

    Adds to atom attributes to the graph *in-place*,
    thus changing the original input graph to include depiction coordinates.

    :param graph: The graph to calculate depiction coordinates for.
    :type graph: nx.Graph
    """
    # Makes list of non-hydrogen, non-terminal atoms in the graph
    atoms: List[int] = list(graph.nodes)
    prune_hydrogens(graph, atoms)
    prune_terminals(graph, atoms)

    constraints_candidates: ConstraintsCandidates = {}

    # Determine chain constraints
    chain_constraints = _find_chain_constraints(atoms, graph)
    constraints_candidates.update(chain_constraints)

    # Determine atom constraints
    for atom_index in atoms:
        patterns = _find_atom_constraints(atom_index, graph)
        if not patterns:
            raise DepicterError(
                f"No candidate constraints found for atom with index {atom_index}"
            )
        constraints_candidates[(atom_index,)] = (
            [(neighbor_constraints,) for neighbor_constraints, _ in patterns],
            [weight for _, weight in patterns],
        )

    # Produce constraint samples
    samples: List[DepictionConstraints] = []
    for _ in range(SAMPLE_SIZE):
        sample = DepictionConstraints()
        if _sample_constraints(sample, constraints_candidates):
            samples.append(sample)
    if not samples:
        raise DepicterError("Could not satisfy constraints")
    # Convert constraints to Cartesian coordinates
    coordinates_samples_with_weights = [
        (_apply_depiction_sample(graph, sample), sample.weights) for sample in samples
    ]
    best_sample = _choose_best_sample(coordinates_samples_with_weights, graph)
    # Postprocess constraints
    _postprocess_sample(best_sample)
    for atom_index, coords in best_sample.items():
        set_depict_coords(atom_index, graph, coords)
