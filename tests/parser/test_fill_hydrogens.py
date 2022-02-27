#!/usr/bin/env python3

"""
tests.parser.test_fill_hydrogens

Tests the calculation of the hcount of organic subset atoms.
"""

from typing import Dict, List, Tuple

import networkx as nx
import pytest

from pydepict.consts import Atom, Bond
from pydepict.parser import fill_hydrogens, new_atom, new_bond

GraphData = Tuple[List[Atom], Dict[int, Dict[int, Bond]], Dict[int, int]]


# Graph data is a tuple of three lists: a list of elements;
# a dictionary (keyed by atom index of the first list) of dictionaries (keyed
# again by atom index to denote graph adjacency) of bond orders;
# and a dictionary mapping atom index to expected hcounts for the organic molecules
@pytest.fixture(
    scope="module",
    params=[
        (  # Methane
            ["C"],
            {},
            {0: 4},
        ),
        (  # Ethane
            ["C", "C"],
            {
                0: {1: 1},
            },
            {0: 3, 1: 3},
        ),
        (  # Ethene
            ["C", "C"],
            {
                0: {1: 2},
            },
            {0: 2, 1: 2},
        ),
        (  # Propane
            ["C", "C", "C"],
            {
                0: {1: 1},
                1: {2: 1},
            },
            {0: 3, 1: 2, 2: 3},
        ),
        (  # Propene
            ["C", "C", "C"],
            {
                0: {1: 2},
                1: {2: 1},
            },
            {0: 2, 1: 1, 2: 3},
        ),
        (  # Bromoethane
            ["C", "C", "Br"],
            {
                0: {1: 1},
                1: {2: 1},
            },
            {0: 3, 1: 2, 2: 0},
        ),
        (  # Ammonia
            ["N"],
            {},
            {0: 3},
        ),
    ],
)
def graph_data(request: pytest.FixtureRequest) -> GraphData:
    return request.param


@pytest.fixture(scope="module")
def graph(graph_data: GraphData) -> nx.Graph:
    g = nx.Graph()
    for atom_index, element in enumerate(graph_data[0]):
        g.add_node(atom_index, **new_atom(element=element, hcount=None))
    for u, edges in graph_data[1].items():
        for v, order in edges.items():
            g.add_edge(u, v, **new_bond(order=order))

    return g


@pytest.fixture(scope="module")
def expected_hcounts(graph_data: GraphData) -> Dict[int, int]:
    return graph_data[2]


def test_fill_hydrogens(graph: nx.Graph, expected_hcounts: Dict[int, int]):
    graph = graph.copy()
    fill_hydrogens(graph)
    for atom_index, hcount in expected_hcounts.items():
        assert graph.nodes[atom_index]["hcount"] == hcount
