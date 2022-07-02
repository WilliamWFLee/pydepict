#!/usr/bin/env python3

"""
pydepict.types

Type aliases.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details.
"""

from typing import Dict, List, Optional, Set, Tuple, Union

from .models import Vector

ChiralSpec = Optional[Tuple[Optional[str], int]]
BondAttribute = Optional[Union[bool, str, int, float]]
AtomAttribute = Optional[Union[bool, str, int, float, ChiralSpec]]
Atom = Dict[str, AtomAttribute]
Bond = Dict[str, BondAttribute]
ParserChain = Tuple[List[Atom], List[Bond]]
Rnum = Tuple[int, Optional[float]]
Rnums = Dict[int, Rnum]
NeighborSpec = Tuple[Optional[str], Optional[float]]
NeighborPattern = Set[Tuple[NeighborSpec, Vector]]
NeighborVectors = Dict[int, Vector]
AtomRnums = List[Tuple[int, Optional[float]]]
AtomPatternsDict = Dict[Optional[str], List[Tuple[NeighborPattern, float]]]
ChainPattern = Tuple[Tuple[Vector, Vector], Dict[int, Tuple[Vector, ...]]]
ConstraintsCandidates = Dict[
    Tuple[int],
    Tuple[
        List[List[NeighborVectors]],
        List[float],
    ],
]
GraphCoordinates = Dict[int, Vector]
DepicterChain = List[int]
