#!/usr/bin/env python3

"""
pydepict.renderer

Renderer for molecular graphs with relative Cartesian coordinates.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

from functools import wraps
from threading import RLock, Thread
from typing import Optional

import networkx as nx
import pygame

from .consts import (
    BLACK,
    DISPLAY_BOND_LENGTH,
    FONT_SIZE,
    FRAME_MARGIN,
    LINE_WIDTH,
    TEXT_MARGIN,
    WHITE,
)
from .utils import (
    average_depicted_bond_length,
    get_depict_coords,
    get_display_coords,
    set_display_coords,
)

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
        self._display_lock = RLock()
        self.graph = graph

    def _with_display_lock(meth):
        """
        Decorator for methods that acquires the display lock
        before calling the wrapped method, and then releases it
        once the wrapped method returns.
        """

        @wraps(meth)
        def wrapper(self: "Renderer", *args, **kwargs):
            with self._display_lock:
                return meth(self, *args, **kwargs)

        return wrapper

    @property
    def graph(self):
        """
        The molecular graph rendered by this renderer instance.

        Setting this property changes the diagram displayed by the renderer.
        """
        return self._graph

    @graph.setter
    @_with_display_lock
    def graph(self, graph: Optional[nx.Graph]):
        self._graph = None if graph is None else graph.copy()
        self._calculate_geometry()

    def _calculate_geometry(self):
        # Calculates display coordinates for atoms in the graph,
        # and recalculates the required display size.

        # Calculate scale factor from depiction coordinates to display coordinates
        if self._graph is not None:
            if self._graph.edges:
                average_bond_length = average_depicted_bond_length(self._graph)
                scale_factor = DISPLAY_BOND_LENGTH / average_bond_length
            else:
                scale_factor = 1

            # Normalises depiction coordinates to be non-negative
            min_x = min((n[1] for n in self._graph.nodes(data="x")), default=0)
            min_y = min((n[1] for n in self._graph.nodes(data="y")), default=0)
            for atom_index in self._graph.nodes:
                self._graph.nodes[atom_index]["x"] -= min_x
                self._graph.nodes[atom_index]["y"] -= min_y

            # Calculates display coordinates, adding margin
            for atom_index in self._graph.nodes:
                set_display_coords(
                    atom_index,
                    self._graph,
                    tuple(
                        v * scale_factor + FRAME_MARGIN
                        for v in get_depict_coords(atom_index, self._graph)
                    ),
                )

            # Calculates display size
            max_dx = max((n[1] for n in self._graph.nodes(data="dx")), default=0)
            max_dy = max((n[1] for n in self._graph.nodes(data="dy")), default=0)
        else:
            max_dx = max_dy = 0
        with self._display_lock:
            self._display = pygame.display.set_mode(
                (max_dx + FRAME_MARGIN, max_dy + FRAME_MARGIN)
            )

    @_with_display_lock
    def _render_bonds(self):
        for u, v in self._graph.edges:
            coords1 = get_display_coords(u, self._graph)
            coords2 = get_display_coords(v, self._graph)

            pygame.draw.line(self._display, BLACK, coords1, coords2, LINE_WIDTH)

    @_with_display_lock
    def _render_atoms(self):
        for atom_index in self._graph.nodes:
            # Skip if element is carbon
            element = self._graph.nodes[atom_index]["element"]
            if element == "C":
                continue
            # Render text from font
            text = self._font.render(element, True, BLACK)
            # Blit text onto white background with margin
            margined_text = pygame.Surface(
                (
                    text.get_width() + TEXT_MARGIN * 2,
                    text.get_height() + TEXT_MARGIN * 2,
                ),
            )
            margined_text.fill(WHITE)
            margined_text.blit(text, (TEXT_MARGIN, TEXT_MARGIN))
            # Blit margined text onto canvas, anchored at the center of the text
            x, y = get_display_coords(atom_index, self._graph)
            coords = (
                x - text.get_width() / 2 - TEXT_MARGIN,
                y - text.get_height() / 2 - TEXT_MARGIN,
            )
            self._display.blit(margined_text, coords)

    @_with_display_lock
    def _render(self):
        self._display.fill(WHITE)
        self._render_bonds()
        self._render_atoms()
        pygame.display.update()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                break

    def _init(self):
        pygame.init()
        self._calculate_geometry()
        self._running = True
        self._font = pygame.font.SysFont(pygame.font.get_default_font(), size=FONT_SIZE)

    def _loop(self):
        while self._running:
            self._handle_events()
            if not self._running:
                break
            self._render()

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
        # pygame quits when the current event loop iteration is completed
        self._running = False
