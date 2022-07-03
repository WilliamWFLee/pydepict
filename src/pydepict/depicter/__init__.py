#!/usr/bin/env python3

"""
pydepict.depicter

Depicter for determining the graphical placement of atoms in a molecular graph.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details.
"""

import random
from copy import deepcopy
from itertools import combinations, cycle
from typing import Dict, Generator, List, Optional, Tuple

import networkx as nx

from ..consts import THIRTY_DEGS_IN_RADS
from ..errors import DepicterError
from ..models import Matrix, Vector
from ..types import (
    ConstraintsCandidates,
    DepicterChain,
    GraphCoordinates,
    NeighborVectors,
)
from ..utils import (
    depiction_width,
    get_atom_attrs,
    neighbors,
    num_bond_order,
    num_heavy_atom_neighbors,
    prune_hydrogens,
    prune_terminals,
)
from .consts import (
    ATOM_PATTERNS,
    CHAIN_ELEMENTS,
    CHAIN_PATTERN_UNITS,
    DEPICTION_SAMPLE_SIZE,
    EPSILON,
    MAX_DEPICTION_SAMPLE_ATTEMPTS,
)
from .models import DepictionConstraints

__all__ = ["depict"]


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


def find_atom_constraints(
    atom_index: int,
    graph: nx.Graph,
) -> List[Tuple[NeighborVectors, float]]:
    """
    Retrieves all possible atom constraints
    for the atom with the specified index in the specified graph.

    :param atom_index: The index of the atom to identify constraints for.
    :type atom_index: int
    :param graph: The graph to use to retrieve molecular graph information.
    :type graph: nx.Graph
    :return: A list of 2-tuples, the first element of each is a dictionary
             of vectors mapping the neighbor index to the vector
             from the atom to the neighbor of that index,
             and the second element is the sampling weight.
    """
    # Determines element of atom
    element = graph.nodes[atom_index]["element"]
    # Get neighbor indices
    neighbor_idxs = tuple(graph[atom_index])
    if not neighbor_idxs:
        # Atom has no neighbors
        return [({}, 1)]
    # Determine bond specifications
    neighbor_specs = tuple(
        (graph.nodes[u]["element"], graph[u][atom_index]["order"])
        for u in neighbor_idxs
    )
    patterns = ATOM_PATTERNS[element if element in ATOM_PATTERNS else None]
    # Determine candidates
    candidates: List[Tuple[Tuple[Vector], float]] = []
    # Iterate over possible patterns
    for pattern in patterns:
        vectors_list = pattern.match(neighbor_idxs, neighbor_specs)
        if vectors_list:
            candidates.extend((vectors, pattern.weight) for vectors in vectors_list)

    return [
        ({idx: vector for idx, vector in zip(neighbor_idxs, vectors)}, weight)
        for vectors, weight in candidates
    ]


def find_longest_chain_from(
    start: int, chain_atoms: List[int], graph: nx.Graph
) -> DepicterChain:
    """
    Finds the longest chain starting from the specified atom index.

    :param start: The atom to start from.
    :type start: int
    :param chain_atoms: A list of chains atoms that can be used
                        for forming new chains.
    :type chain_atoms: List[int]
    :param graph: The graph to use to retrieve molecular graph information.
    :type graph: nx.Graph
    """
    start_neighbors = (
        atom_index
        for atom_index in graph[start]
        if is_chain_atom(atom_index, graph) and atom_index in chain_atoms
    )
    chains = []
    for neighbor_idx in start_neighbors:
        next_chain_atoms = chain_atoms.copy()
        next_chain_atoms.remove(start)
        chains.append(find_longest_chain_from(neighbor_idx, next_chain_atoms, graph))
    if not chains:
        return []
    return [start] + max(chains, key=lambda x: len(x))


def find_chains(chain_atoms: List[int], graph: nx.Graph) -> List[DepicterChain]:
    """
    Finds all chains in a graph.

    This is a greedy algorithm that is not guaranteed
    to find the optimal set of longest chains.

    :param atoms: A list of chain atom indices that are can be used for chaining.
    :type atoms: List[int]
    :param graph: The molecular graph to retrieve chemical information from.
    :type graph: nx.Graph
    :return: A list of lists, each inner list being a single chain,
             represented as a list of atom indices.
    :rtype: List[List[int]]
    """
    chains = []
    while len(chain_atoms) > 4:
        chain = find_longest_chain_from(chain_atoms[0], chain_atoms, graph)
        if not chain:
            break
        for atom_index in chain:
            chain_atoms.remove(atom_index)
        chains.append(chain)

    return chains


def chain_neighbors(
    chain: DepicterChain, graph: nx.Graph
) -> Generator[Tuple[int, int, Tuple[int, ...]], None, None]:
    """
    Determines the neighbors of each atom in a chain.

    Returns a generator for iterating over each set of neighbors.

    :param chain: The chain to determine the neighbors for.
    :type chain: DepicterChain
    :param graph: The graph associated with the chain.
    :type graph: nx.Graph
    :return: A generator for iterating over the neighbors of each atom,
             producing a 3-tuple: the first element
             the index of the preceding atom in the chain; the second element
             the index of the next atom in the chain; and the third element
             a tuple of the other substituents of the chain atom.
    :rtype: Generator[Tuple[int, int, Tuple[int, ...]], None, None]
    """

    def find_other_chain_endpoint(chain_neighbor: int) -> Optional[int]:
        other_chain_atoms = [
            v
            for v in root_neighbors
            if v != chain_neighbor and graph.nodes[v]["element"] in CHAIN_ELEMENTS
        ]
        if other_chain_atoms:
            return other_chain_atoms[0]
        return None

    def find_chain_subs(chain_neighbors: Tuple[int, int]) -> Tuple[int, ...]:
        subs = tuple(
            neighbor for neighbor in root_neighbors if neighbor not in chain_neighbors
        )
        return subs

    root_neighbors = neighbors(chain[0], graph)
    left = find_other_chain_endpoint(chain[1])
    yield left, chain[1], find_chain_subs((left, chain[1]))

    for left, root, right in zip(chain[:-2], chain[1:-1], chain[2:]):
        root_neighbors = neighbors(root, graph)
        yield left, right, find_chain_subs((left, right))

    root_neighbors = neighbors(chain[-1], graph)
    right = find_other_chain_endpoint(chain[-2])
    yield chain[-2], right, find_chain_subs((chain[-2], right))


def find_chain_constraints(atoms: List[int], graph: nx.Graph) -> ConstraintsCandidates:
    """
    Finds chains within the molecular graph, and returns a set of constraints
    for those chains.

    :param atoms: A list of available atom indices for chains.
    :type atoms: List[int]
    :param graph: The graph to retrieve molecular graph information from.
    :type graph: nx.Graph
    :return: A dictionary of constraints candidates.
    :rtype: ConstraintsCandidates.
    """
    chains = find_chains([atom for atom in atoms if is_chain_atom(atom, graph)], graph)
    candidates: ConstraintsCandidates = {}
    for chain in chains:
        patterns = []
        for pattern_units in CHAIN_PATTERN_UNITS:
            pattern = []
            for (left, right, subs), chain_pattern in zip(
                chain_neighbors(chain, graph), cycle(pattern_units)
            ):
                neighbor_constraints = {}
                for neighbor_idx, vector in zip((left, right), chain_pattern[0]):
                    if neighbor_idx is not None:
                        neighbor_constraints[neighbor_idx] = vector
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


def remove_conflicting_constraints(
    endpoints: Tuple[int, int],
    vector: Vector,
    constraints_candidates: ConstraintsCandidates,
) -> bool:
    """
    Removes conflicting constraints candidates from a constraints candidates dictionary,
    given a chosen vector between two nodes, and returns a boolean indicating
    if there is at least one constraint candidate left for every block.

    :param endpoints: The nodes between which the vector has been fixed,
    :type endpoints: Tuple[int, int]
    :param vector: The vector between :param:`endpoints`, from the first element
                   to the second element.
    :type vector: Vector
    :param constraints_candidates: The dictionary of constraints candidates
                                   that is searched in order to find conflicting
                                   candidate constraints.
    :type constraints_candidates: ConstraintsCandidates.
    :return: A boolean indicating whether there exists
             at least one constraints candidate for every block.
    :rtype: bool
    """
    u, v = endpoints
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


def sample_constraints(
    constraints_candidates: ConstraintsCandidates,
) -> Optional[DepictionConstraints]:
    """
    Samples constraints from a constraints candidates dictionary,
    and returns a new depiction constraints sample.

    :param constraints_candidates: The dictionary of constraints candidates
                                   to sample constraints from.
    :type constraints_candidates: ConstraintsCandidates
    :return: The constraints sample produced, or :data:`None` if sampling failed.
    :rtype: Optional[DepictionConstraints]
    """
    sample = DepictionConstraints()
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
                constraints_left = remove_conflicting_constraints(
                    (u, v), vector, candidates_copy
                )
                if not constraints_left:
                    return None
    return sample


def apply_depiction_sample(
    graph: nx.Graph, constraints: DepictionConstraints
) -> GraphCoordinates:
    """
    Applies constraints to a graph to produce a dictionary mapping atom index
    to the position vector of that atom.

    :param graph: The graph to use to perform a depth-first tree traversal
    :type graph: nx.Graph
    :param constraints: The constraints to apply in order to build
                        the dictionary of coordinates.
    :type constraints: DepictionConstraints
    :return: The dictionary of atom positions.
    :rtype: GraphCoordinates
    """
    coordinates = {0: Vector(0, 0)}
    for u, v in nx.dfs_edges(graph, source=0):
        coordinates[v] = coordinates[u] + constraints[u, v]

    return coordinates


def depiction_coordinates_congestion(
    coordinates: GraphCoordinates,
    weights: Dict[int, float],
    graph: nx.Graph,
) -> float:
    """
    Calculates the congestion from the set of coordinates for a sample,
    and the graph that it describes.

    :param coordinates: The dictionary of depiction coordinates.
    :type coordinates: GraphCoordinates
    :param weights: A dictionary mapping each atom to the weight of the constraint
                    that determined its placement.
    :type weights: Dict[int, float]
    :return: The congestion as a float:
             the higher the number, the highest the congestion.
    :rtype: float
    """
    congestion = 0
    for component in nx.connected_components(graph):
        for u, v in combinations(component, 2):
            if not graph.has_edge(u, v):
                congestion += 1 / (
                    (Vector.distance(coordinates[u], coordinates[v]) + EPSILON) ** 2
                    * weights[u]
                    * weights[v]
                )

    return congestion


def select_best_sample(
    samples_with_weights: List[Tuple[GraphCoordinates, Dict[int, float]]],
    graph: nx.Graph,
) -> GraphCoordinates:
    """
    Selects the best depiction sample from a list of dictionaries
    of coordinate samples, and returns it.

    :param samples_with_weights: A list of 2-tuples coordinate samples with weights.
                                 The first element of each tuple is a graph coordinates
                                 dictionary; the second element is a sampling weight
                                 dictionary.
    :type samples_with_weights: List[Tuple[GraphCoordinates], Dict[int, float]]]
    :return: The best sample, as determined by congestion.
    :rtype: GraphCoordinates
    """
    best_sample, _ = min(
        samples_with_weights,
        key=lambda sample_with_weight: depiction_coordinates_congestion(
            *sample_with_weight, graph
        ),
    )
    return best_sample


def maximize_sample_width(sample: GraphCoordinates):
    """
    Rotates a depiction sample in place such that its width is maximized.

    :param sample: The sample whose width should be maximised.
    :type sample: GraphCoordinates
    """
    matrices = [Matrix.rotate(THIRTY_DEGS_IN_RADS * i) for i in range(12)]
    widest_sample = max(
        (
            {atom_index: matrix * vector for atom_index, vector in sample.items()}
            for matrix in matrices
        ),
        key=depiction_width,
    )
    sample.update(widest_sample)


def postprocess_sample(sample: GraphCoordinates):
    """
    Postprocesses a depiction sample to produce the final depiction.

    :param sample: The dictionary of coordinates to postprocess
    :type sample: GraphCoordinates
    """
    maximize_sample_width(sample)


def depict(graph: nx.Graph) -> GraphCoordinates:
    """
    Determines depiction coordinates for the graph, and returns them.

    :param graph: The graph to calculate depiction coordinates for.
    :type graph: nx.Graph
    :return: A dictionary mapping atom index to position vector
    :rtype: GraphCoordinates
    """
    # Makes list of non-hydrogen, non-terminal atoms in the graph
    atoms: List[int] = list(graph.nodes)
    prune_hydrogens(graph, atoms)
    prune_terminals(graph, atoms)

    constraints_candidates: ConstraintsCandidates = {}

    # Determine chain constraints
    chain_constraints = find_chain_constraints(atoms, graph)
    constraints_candidates.update(chain_constraints)

    # Determine atom constraints
    for atom_index in atoms:
        patterns = find_atom_constraints(atom_index, graph)
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
    attempts = 0
    while (
        len(samples) < DEPICTION_SAMPLE_SIZE
        and attempts < MAX_DEPICTION_SAMPLE_ATTEMPTS
    ):
        sample = sample_constraints(constraints_candidates)
        if sample is not None and sample not in samples:
            samples.append(sample)
        attempts += 1

    if not samples:
        raise DepicterError("Could not satisfy constraints")
    # Convert constraints to Cartesian coordinates
    coordinates_samples_with_weights = [
        (apply_depiction_sample(graph, sample), sample.weights) for sample in samples
    ]
    best_sample = select_best_sample(coordinates_samples_with_weights, graph)
    # Postprocess constraints
    postprocess_sample(best_sample)
    return best_sample
