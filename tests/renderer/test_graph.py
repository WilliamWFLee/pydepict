#!/usr/bin/env python3

"""
tests.renderer.test_dev

Script for displaying test molecules for development purposes
"""

import time
from typing import Tuple

import networkx as nx
import pytest

from pydepict.consts import GraphCoordinates
from pydepict.models import Vector
from pydepict.renderer import Renderer

from .utils import requires_video


@pytest.fixture
def structure1() -> Tuple[nx.Graph, GraphCoordinates]:
    graph = nx.Graph()
    graph.add_node(0, element="C")
    graph.add_node(1, element="C")
    graph.add_node(2, element="O")
    graph.add_edge(0, 1, order=1)
    graph.add_edge(1, 2, order=1)

    positions = {
        0: Vector(0, 0),
        1: Vector(1, 1),
        2: Vector(2, 0),
    }

    return graph, positions


@pytest.fixture
def structure2() -> Tuple[nx.Graph, GraphCoordinates]:
    graph = nx.Graph()
    graph.add_node(0, element="C")
    graph.add_node(1, element="C")
    graph.add_edge(0, 1, order=1)

    positions = {
        0: Vector(0, 0),
        1: Vector(-1, 1),
    }

    return graph, positions


@requires_video
def test_blocking_renderer(structure1: Tuple[nx.Graph, GraphCoordinates]):
    renderer = Renderer(*structure1)
    renderer.show()


@requires_video
def test_nonblocking_renderer(
    structure1: Tuple[nx.Graph, GraphCoordinates],
    structure2: Tuple[nx.Graph, GraphCoordinates],
):
    with Renderer(*structure1) as renderer:
        time.sleep(3)
        renderer.set_structure(*structure2)
        time.sleep(3)
