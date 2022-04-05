#!/usr/bin/env python3

"""
pydepict

A library for parsing and display SMILES strings.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""


from typing import NamedTuple, Optional

from .depicter import depict
from .parser import parse
from .renderer import render

__all__ = [
    "__author__",
    "__copyright__",
    "__license__",
    "version_info",
    "__version__",
    "show",
    "parse",
    "depict",
    "render",
]


class _VersionInfo(NamedTuple):
    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None

    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}" + (
            f"-{self.pre_release}" if self.pre_release is not None else ""
        )


__author__ = "William Lee; The University of Sheffield"
__copyright__ = "Copyright (c) 2022 William Lee and The University of Sheffield"
__license__ = "MIT"

version_info = _VersionInfo(0, 1, 0)
__version__ = str(version_info)


def show(smiles: str):
    """
    Parses and displays a diagram representing the input SMILES string.

    :param smiles: The SMILES string to parse and display
    :type smiles: str
    """
    graph, _ = parse(smiles)
    depict(graph)
    render(graph)


# Clean up namespace
del NamedTuple, Optional
