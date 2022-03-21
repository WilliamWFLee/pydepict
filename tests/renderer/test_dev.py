#!/usr/bin/env python3

"""
tests.renderer.test_dev

Script for displaying test molecules for development purposes
"""

import networkx as nx
import pytest

from pydepict.renderer import Renderer


@pytest.fixture
def graph():
    g = nx.Graph()
    g.add_node(0, element="C", x=0, y=0)
    g.add_node(1, element="C", x=2, y=2)
    g.add_edge(0, 1)

    return g


@pytest.fixture
def renderer(graph: nx.Graph):
    return Renderer(graph)


def test_renderer(renderer: Renderer):
    renderer.show()
