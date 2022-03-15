#!/usr/bin/env python3

"""
pydepict.renderer

Renderer for molecular graphs with relative Cartesian coordinates.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

from threading import RLock, Thread
from typing import Optional

import networkx as nx
import pygame

from .consts import WHITE

__all__ = ["Renderer"]


class Renderer:
    """
    Renderer class for rendering molecular graphs.

    The renderer takes a molecular graph, where each node in the graph
    has been assigned a pair of Cartesian coordinates. These coordinates are then
    scaled and/or translated into coordinates on a graphics canvas,
    and then the molecule is rendered.

    .. attribute:: graph
        Instance of a molecular graph to be rendered by the renderer,
        or :data:`None` for no graph.

        :type: Optional[nx.Graph]
    """

    def __init__(self, graph: Optional[nx.Graph] = None):
        self.graph = graph
        self._graph_lock = RLock()

    @property
    def graph(self):
        """
        The molecular graph rendered by this renderer instance.

        Setting this property changes the diagram displayed by the renderer.
        """
        return self._graph

    @graph.setter
    def graph(self, graph: Optional[nx.Graph]):
        self._graph = None if graph is None else graph.copy()   

    def _init(self):
        pygame.init()
        self._screen = pygame.display.set_mode((800, 600))
        self._running = True

    def _loop(self):
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    break
            if not self._running:
                break
            self._screen.fill(WHITE)
            pygame.display.update()

        pygame.quit()

    def show(self, blocking: bool = True):
        """
        Displays the renderer window.

        This method blocks the calling thread with the event loop,
        unless :param:`blocking` is set to :data:`True`, in which case
        the event loop is called in a separate thread, and the method
        returns after the thread is started.

        :param blocking: Whether or not this method blocks the calling thread.
        :type blocking: bool
        """
        self._init()
        if blocking:
            self._loop()
        t = Thread(target=self._loop, daemon=True)
        t.start()

    def close(self):
        """
        Closes the renderer window.
        """
        self._running = False
