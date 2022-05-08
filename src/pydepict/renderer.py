#!/usr/bin/env python3

"""
pydepict.renderer

Renderer for molecular graphs with relative Cartesian coordinates.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

import os
from collections import defaultdict
from copy import deepcopy
from functools import wraps
from math import sqrt
from threading import RLock, Thread
from typing import Any, Callable, Optional

import networkx as nx
import pygame

from .consts import (
    BLACK,
    BOND_WIDTH,
    DISPLAY_BOND_LENGTH,
    FONT_FAMILY,
    FONT_SIZE,
    FRAME_MARGIN,
    SCREENSHOTS_DIR,
    TEXT_MARGIN,
    WHITE,
    WINDOW_TITLE,
    GraphCoordinates,
)
from .models import Vector
from .utils import average_depicted_bond_length, get_datetime_filename

__all__ = ["Renderer", "render"]


class Renderer:
    """
    Renderer class for rendering molecular graphs.

    The renderer takes a molecular graph, and a dictionary of depiction coordinates.
    These coordinates are then scaled and/or translated
    into coordinates on a graphics canvas, then the molecule is rendered.

    .. attribute:: graph

        Instance of a molecular graph to be rendered by the renderer,
        or :data:`None` for no graph.

        :type: Optional[nx.Graph]

    .. attribute:: render_positions

        Dictionary of render positions for each atom in the graph.

        :type: GraphCoordinates

    .. attribute:: redraw

        Whether or not the diagram should be redrawn on the next event loop iteration.
        Set this to :data:`true` whenever any of the depiction coordinates
        in :attr:`graph` are changed, but you do not set :attr:`graph` itself.

        :type: bool
    """

    def __init__(
        self,
        graph: Optional[nx.Graph] = None,
        positions: Optional[GraphCoordinates] = None,
        *,
        title: Optional[str] = None,
    ):
        self._display_lock = RLock()
        self.render_positions: Optional[GraphCoordinates] = None
        self._text_radii = {}
        self.redraw = False  # XXX: Must be before setting graph
        self.title = title
        self._thread = None
        self._event_cbs = defaultdict(lambda: set())
        self.set_structure(graph, positions)

    def __enter__(self) -> "Renderer":
        self.show(False)
        return self

    def __exit__(self, exc_type, exc_val, exc_cb):
        self.close()

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

    @_with_display_lock
    def set_structure(
        self,
        graph: Optional[nx.Graph] = None,
        depiction_positions: Optional[GraphCoordinates] = None,
    ):
        """
        Sets the graph and corresponding depiction positions dictionary
        for the renderer

        :param graph: The graph that the renderer displays, defaults to :data`None`
        :type graph: Optional[nx.Graph]
        :param depiction_positions: The depiction positions of the graph to render,
                                    defaults to :data:`None`
        :type depiction_positions: Optional[GraphCoordinates]
        """
        self._graph = graph
        self.set_positions(depiction_positions)

    @_with_display_lock
    def set_positions(self, depiction_positions: Optional[GraphCoordinates] = None):
        """
        Sets the depiction positions dictionary of the graph.

        :param depiction_positions: The dictionary of depiction positions,
                                    defaults to :data:`None'
        :type depiction_positions: Optional[GraphCoordinates]
        """
        self._calculate_geometry(depiction_positions)
        self.redraw = True

    @property
    def graph(self) -> nx.Graph:
        """
        The graph that is being rendered. This is a read-only property.
        Use :meth:`set_structure` to set the graph and depiction coordinates.
        """
        return self._graph

    @property
    def title(self) -> str:
        """
        The title to be displayed in the renderer window.

        Setting this property changes the title shown by the window manager.
        """
        return self._title

    @title.setter
    def title(self, title: Optional[str] = None):
        self._title = title
        pygame.display.set_caption(
            (f"{self._title} - " if self._title is not None else "") + WINDOW_TITLE
        )

    def _calculate_geometry(
        self, depiction_positions: Optional[GraphCoordinates] = None
    ):
        # Calculates display coordinates for atoms in the graph,
        # and recalculates the required display size.
        if self._graph is not None and depiction_positions is None:
            raise ValueError(
                "Depiction coordinates cannot be None if graph is not None"
            )

        if self._graph is not None:
            for atom_index in self._graph.nodes:
                if atom_index not in depiction_positions:
                    raise ValueError(f"Position for atom {atom_index} not provided")
            # Calculate scale factor from depiction coordinates to display coordinates
            if self._graph.edges:
                average_bond_length = average_depicted_bond_length(
                    self._graph, depiction_positions
                )
                scale_factor = DISPLAY_BOND_LENGTH / average_bond_length
            else:
                scale_factor = 1

            # Copy depiction positions to render positions
            self.render_positions = deepcopy(depiction_positions)
            # Invert y-coordinate as graphics origin is in top-left hand corner
            for atom_index in self.render_positions:
                self.render_positions[atom_index] *= Vector(1, -1)

            min_all_render_vector = Vector.min_all(self.render_positions.values())
            for atom_index in self._graph.nodes:
                # Normalises depiction coordinates to be non-negative
                self.render_positions[atom_index] -= min_all_render_vector
                # Multiplies by scale factor
                self.render_positions[atom_index] *= scale_factor
                # Adds frame margin
                self.render_positions[atom_index] += Vector(FRAME_MARGIN, FRAME_MARGIN)

            # Calculates display size
            max_all_render_vector = Vector.max_all(self.render_positions.values())
        else:
            self.render_positions = None
            max_all_render_vector = Vector(0, 0)
        self.display_size = tuple(
            max_all_render_vector + Vector(FRAME_MARGIN, FRAME_MARGIN)
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
            x, y = self.render_positions[atom_index]
            coords = (x - text.get_width() / 2, y - self._font.get_ascent() / 2)
            self.display.blit(text, coords)
            # Store radius of rendered text
            self._text_radii[atom_index] = (
                sqrt(text.get_width() ** 2 + self._font.get_height() ** 2) / 2
                + TEXT_MARGIN
            )
        else:
            # Set atom display radius to 0
            self._text_radii[atom_index] = 0

    @_with_display_lock
    def _render_bond(self, u: int, v: int):
        bond_order = self._graph.edges[u, v]["order"]
        # Coordinates for bond endpoints
        coords1, coords2 = self.render_positions[u], self.render_positions[v]
        # Retrieve rendered radius for atoms, including margin
        atom_radius1, atom_radius2 = self._text_radii[u], self._text_radii[v]
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
            pygame.draw.aaline(
                self.display, BLACK, line_end1 + offset, line_end2 + offset
            )

    @_with_display_lock
    def _render(self):
        if self.redraw:
            self._text_radii.clear()
            self.display = pygame.display.set_mode(self.display_size)
            # Draw on display
            self.display.fill(WHITE)
            if self._graph is not None:
                for atom_index in self._graph.nodes:
                    self._render_atom(atom_index)
                for u, v in self._graph.edges:
                    self._render_bond(u, v)
            self.redraw = False
        pygame.display.update()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                break
            elif event.type == pygame.KEYDOWN:
                ctrl = pygame.key.get_mods() & pygame.KMOD_CTRL
                if ctrl and event.key == pygame.K_s:
                    pygame.image.save(
                        self.display,
                        os.path.join(SCREENSHOTS_DIR, get_datetime_filename() + ".png"),
                    )

    def _init(self):
        pygame.init()
        self._running = True
        self._font = pygame.font.SysFont(FONT_FAMILY, size=FONT_SIZE)

    def _loop(self):
        while self._running:
            self._handle_events()
            if not self._running:
                break
            self._render()

        pygame.quit()
        self._fire("close")

    def _fire(self, event: str):
        """
        Fire callbacks for :param:`event`.
        """
        for func in self._event_cbs[event]:
            func()

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
        else:
            self._thread = Thread(target=self._loop, daemon=True)
            self._thread.start()

    def on(self, event: str, func: Callable[[], Any]):
        """
        Adds a callback function that is called when event
        with name :param:`event` is fired, e.g.::
            renderer.on('close', callback)

        The callback function should have no required arguments,
        and any return value is ignored.

        :param event: The name of the event to bind a callback for
        :type event: str
        :param func: The callback function to add.
        :type func: Callable[[], Any]
        """
        self._event_cbs[event].add(func)

    def not_on(self, event: str, func: Callable[[], Any]):
        """
        Removes a callback function from being called
        when an event with name :param:`event` is fired.

        If the callback function does not exist, then this is ignored.

        :param event: The name of the event to remove
        :type event: str
        :param func: The callback function to remove.
        :type func: Callable[[], Any]
        """
        if func in self._event_cbs[event]:
            self._event_cbs[event].remove(func)

    def close(self):
        """
        Closes the renderer window.
        """
        # pygame quits when the current event loop iteration is completed
        self._running = False
        if self._thread is not None:
            self._thread.join()


def render(
    graph: nx.Graph,
    positions: GraphCoordinates,
    *,
    title: Optional[str] = None,
):
    """
    Shortcut for using :class:`Renderer`. Equivalent to::
        renderer = Renderer(graph, positions, title=title)
        renderer.show()

    This function blocks the current thread until the renderer window is closed.

    :param graph: The graph to render
    :type graph: nx.Graph
    :param positions: The depiction positions of the graph to render
    :type: GraphCoordinates
    :param title: The title to display in the renderer window, defaults to :data:`None`
    :type title: str
    """
    renderer = Renderer(graph, positions, title=title)
    renderer.show()
