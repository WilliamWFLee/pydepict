#!/usr/bin/env python3

"""
tests.parser.test_resolve_rnums

Tests the resolution of an rnum specification for a specific atom.
"""

from typing import Dict, Iterable, Tuple

import networkx as nx
import pytest

from pydepict.models import Stream
from pydepict.parser import new_atom, resolve_rnums
from pydepict.types import Atom, AtomRnums, Rnums

GraphData = Tuple[Dict[int, Atom], Iterable[Tuple[int, int]]]
RnumData = Tuple[AtomRnums, int, Rnums, GraphData]


@pytest.fixture
def stream() -> Stream:
    return Stream("")


def find_expected_bond_order(
    bond_order: int,
    other_bond_order: int,
    atom_aromatic: bool,
    other_atom_aromatic: bool,
) -> int:
    # Assumes no bond conflict
    if bond_order is None and other_bond_order is None:
        return 1.5 if atom_aromatic and other_atom_aromatic else 1
    if bond_order is not None:
        return bond_order
    if other_bond_order is not None:
        return other_bond_order


@pytest.fixture(
    scope="module",
    params=[
        (  # Forms single-bond ring
            [(1, None)],
            3,
            {1: (0, None)},
            (
                {0: {}, 1: {}, 2: {}, 3: {}},
                [(0, 1), (1, 2), (2, 3)],
            ),
        ),
        (  # No ring formed, first occurrence of rnum
            [(1, None)],
            3,
            {},
            (
                {0: {}, 1: {}, 2: {}, 3: {}},
                [(0, 1), (1, 2), (2, 3)],
            ),
        ),
        (  # Forms ring with one double bond, specified on current atom
            [(1, 2)],
            3,
            {1: (0, None)},
            (
                {0: {}, 1: {}, 2: {}, 3: {}},
                [(0, 1), (1, 2), (2, 3)],
            ),
        ),
        (  # Forms ring with one double bond, specified on previous atom
            [(1, None)],
            3,
            {1: (0, 2)},
            (
                {0: {}, 1: {}, 2: {}, 3: {}},
                [(0, 1), (1, 2), (2, 3)],
            ),
        ),
        (  # Forms a ring, with one left over
            [(1, None), (2, None)],
            3,
            {1: (0, None)},
            (
                {0: {}, 1: {}, 2: {}, 3: {}},
                [(0, 1), (1, 2), (2, 3)],
            ),
        ),
        (  # Forms an aromatic ring
            [(1, None)],
            3,
            {1: (0, None)},
            (
                {
                    0: {"aromatic": True},
                    1: {"aromatic": True},
                    2: {"aromatic": True},
                    3: {"aromatic": True},
                },
                [(0, 1), (1, 2), (2, 3)],
            ),
        ),
        (  # Explicit single bond between aromatic atoms
            [(1, None)],
            11,
            {1: (0, None)},
            (
                {
                    0: {"aromatic": True},
                    1: {"aromatic": True},
                    2: {"aromatic": True},
                    3: {"aromatic": True},
                    4: {"aromatic": True},
                    5: {"aromatic": True},
                    6: {"aromatic": True},
                    7: {"aromatic": True},
                    8: {"aromatic": True},
                    9: {"aromatic": True},
                    10: {"aromatic": True},
                    11: {"aromatic": True},
                },
                [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 0),
                    (6, 7),
                    (7, 8),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (11, 6),
                ],
            ),
        ),
    ],
)
def valid_rnum_data(request: pytest.FixtureRequest) -> RnumData:
    return request.param


@pytest.fixture
def valid_graph(valid_rnum_data: RnumData) -> nx.Graph:
    atoms, bonds = valid_rnum_data[3]
    graph = nx.Graph()
    for index, attrs in atoms.items():
        graph.add_node(index, **new_atom(**attrs))
    for u, v in bonds:
        graph.add_edge(u, v)

    return graph


def test_valid_rnum_spec(
    valid_rnum_data: RnumData,
    valid_graph: nx.Graph,
    stream: Stream,
):
    atom_rnums, atom_index, rnums = valid_rnum_data[:3]

    # Rnums copied to avoid data needed for assertions being destroyed
    rnums_copy = rnums.copy()
    resolve_rnums(atom_rnums, atom_index, rnums_copy, valid_graph, stream)

    for rnum_index, bond_order in atom_rnums:
        if rnum_index in rnums:
            # Paired rnum
            other_atom_index, other_bond_order = rnums[rnum_index]

            expected_bond_order = find_expected_bond_order(
                bond_order,
                other_bond_order,
                valid_graph.nodes[atom_index]["aromatic"],
                valid_graph.nodes[other_atom_index]["aromatic"],
            )
            assert valid_graph.has_edge(atom_index, other_atom_index)
            assert (
                valid_graph[atom_index][other_atom_index]["order"]
                == expected_bond_order
            )
        else:
            # Unpaired rnum
            assert rnum_index in rnums_copy
            assert rnums_copy[rnum_index] == (atom_index, bond_order)
