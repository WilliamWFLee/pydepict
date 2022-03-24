#!/usr/bin/env python3

"""
pydepict.renderer

Renderer for molecular graphs with relative Cartesian coordinates.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

from functools import wraps
from math import sqrt
from threading import RLock, Thread
from typing import Optional, Tuple

import networkx as nx
import pygame

from .consts import (
    BLACK,
    DISPLAY_BOND_LENGTH,
    FONT_FAMILY,
    FONT_SIZE,
    FRAME_MARGIN,
    BOND_WIDTH,
    LINE_WIDTH,
    TEXT_MARGIN,
    WHITE,
)
from .utils import (
    Vector,
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
        self._thread = None

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
        The graph is copied using :meth:`nx.Graph.copy`
        to avoid changing the original graph
        """
        return self._graph

    @graph.setter
    @_with_display_lock
    def graph(self, graph: Optional[nx.Graph]):
        self._graph = None if graph is None else graph.copy()
        self._calculate_geometry()
        self._redraw = True

    def _calculate_geometry(self):
        # Calculates display coordinates for atoms in the graph,
        # and recalculates the required display size.
        if self._graph is not None:
            # Calculate scale factor from depiction coordinates to display coordinates
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
                    Vector(
                        *(
                            v * scale_factor + FRAME_MARGIN
                            for v in get_depict_coords(atom_index, self._graph)
                        )
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

    def _display_atom(self, atom_index: int) -> bool:
        """
        Returns whether to render the atom with the given index
        """
        element = self._graph.nodes[atom_index]["element"]
        if element == "C":
            return False
        return True

    @_with_display_lock
    def _render_atom(self, atom_index: int):
        if self._display_atom(atom_index):
            # Skip rendering if atom should not be displayed
            element = self._graph.nodes[atom_index]["element"]
            # Render text from font
            text = self._font.render(element, True, BLACK)
            # Blit text onto canvas, anchored at the center of the text
            x, y = get_display_coords(atom_index, self._graph)
            coords = (x - text.get_width() / 2, y - self._font.get_ascent() / 2)
            self._display.blit(text, coords)
            # Store radius of rendered text
            self._graph.nodes[atom_index]["dr"] = (
                sqrt(text.get_width() ** 2 + self._font.get_height() ** 2) / 2
                + TEXT_MARGIN
            )
        else:
            # Set atom display radius to 0
            self._graph.nodes[atom_index]["dr"] = 0

    @_with_display_lock
    def _render_bond(self, u: int, v: int):
        bond_order = self._graph.edges[u, v]["order"]
        # Coordinates for bond endpoints
        coords1 = get_display_coords(u, self._graph)
        coords2 = get_display_coords(v, self._graph)
        # Retrieve rendered radius for atoms, including margin
        atom_radius1 = self._graph.nodes[u]["dr"]
        atom_radius2 = self._graph.nodes[v]["dr"]
        # Sort coordinates such that coords1 -> coords2 is left-to-right
        if coords2 < coords1:
            coords1, coords2 = coords2, coords1
        # Get vector between bond endpoints, and its normal
        line_vector = coords2 - coords1
        line_vector_normal = line_vector.normal()
        # Calculate length of bond vector
        atom1_margin_vector = line_vector.scale_to(atom_radius1)
        atom2_margin_vector = line_vector.scale_to(atom_radius2)
        # Calculate ends of bond line
        line_end1 = (coords1 + atom1_margin_vector).floor()
        line_end2 = (coords2 - atom2_margin_vector).floor()

        for i in range(bond_order):
            offset_magnitude = (i / (bond_order - 1) - 1 / 2) if bond_order > 1 else 0
            offset = line_vector_normal.scale_to(offset_magnitude * BOND_WIDTH)
            _aaline(
                self._display,
                BLACK,
                line_end1 + offset,
                line_end2 + offset,
                width=LINE_WIDTH,
            )

    @_with_display_lock
    def _render(self):
        if self._redraw:
            # Draw on display
            self._display.fill(WHITE)
            for atom_index in self._graph.nodes:
                self._render_atom(atom_index)
            for u, v in self._graph.edges:
                self._render_bond(u, v)
            pygame.display.update()
            self._redraw = False

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                break

    def _init(self):
        pygame.init()
        self._calculate_geometry()
        self._running = True
        self._font = pygame.font.SysFont(FONT_FAMILY, size=FONT_SIZE)

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
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self):
        """
        Closes the renderer window.
        """
        # pygame quits when the current event loop iteration is completed
        self._running = False
        if self._thread is not None:
            self._thread.join()


def _aaline(
    surface: pygame.Surface,
    color,
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    width: int = 1,
):
    """
    Render a thick antialiased line
    """
    start_pos = Vector.from_tuple(start_pos)
    end_pos = Vector.from_tuple(end_pos)
    line_vector = end_pos - start_pos
    line_vector_normal = line_vector.normal()
    offset_magnitude = -width / 2
    for _ in range(width):
        offset_vector = line_vector_normal.scale_to(offset_magnitude)
        pygame.draw.aaline(
            surface, color, start_pos + offset_vector, end_pos + offset_vector
        )
        offset_magnitude += 1
