#!/usr/bin/env python3

"""
pydepict.depicter

Depicter for determining the graphical placement of atoms in a molecular graph.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

import networkx as nx

__all__ = ["Depicter", "depict"]


class Depicter:
    """
    Depicter class for depicting molecular graphs.

    A depicter takes a graph representation of a chemical structure,
    and determines the coordinates of each atom in order for the graph
    to be graphically represented.
    """

    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph

    def _prune_hydrogens(self) -> None:
        """
        Remove hydrogens and their adjacent edges from the pruned graph instance.
        """
        for atom_index, element in self._pruned_graph.nodes(data="element"):
            if element == "H":
                self._pruned_graph.remove_node(atom_index)

    def _prune_terminals(self) -> None:
        """
        Remove terminal atoms and their adjacent edges from the pruned graph instance.
        """
        for atom_index in self._pruned_graph.nodes:
            if len(self._pruned_graph[atom_index]) <= 1:
                self._pruned_graph.remove_node(atom_index)

    def depict(self) -> None:
        """
        Determines depiction coordinates for the graph in this depicter.

        Adds to atom attributes to the graph *in-place*,
        thus changing the original input graph to include depiction coordinates.
        """
        self._pruned_graph = self.graph.copy()
        self._prune_hydrogens()
        self._prune_terminals()


def depict(graph: nx.Graph) -> None:
    """
    Shortcut that uses that :class:`Depicter`
    to add depiction coordinates to the input graph.

    It is equivalent to::
        Depicter(graph).depict()
    """
    depicter = Depicter(graph)
    depicter.depict()
