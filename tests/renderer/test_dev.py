#!/usr/bin/env python3

"""
tests.renderer.test_dev

Script for displaying test molecules for development purposes
"""

import time
import networkx as nx
import pytest

from pydepict.renderer import Renderer


@pytest.fixture
def graph1():
    g = nx.Graph()
    g.add_node(0, element="C", x=0, y=0)
    g.add_node(1, element="C", x=1, y=1)
    g.add_edge(0, 1, order=4)

    return g


@pytest.fixture
def graph2():
    g = nx.Graph()
    g.add_node(0, element="C", x=0, y=0)
    g.add_node(1, element="C", x=-1, y=1)
    g.add_edge(0, 1)

    return g


def test_renderer(graph1: nx.Graph):
    renderer = Renderer(graph1)
    renderer.show()


def test_nonblocking_renderer(graph1: nx.Graph, graph2: nx.Graph):
    renderer = Renderer(graph1)
    renderer.show(False)
    time.sleep(3)
    renderer.graph = graph2
    time.sleep(3)
    renderer.close()
