#!/usr/bin/env python3

"""
pydepict.types

Type aliases.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details.
"""

from typing import Dict, List, Optional, Tuple, Union

from .models import Vector

ChiralSpec = Optional[Tuple[Optional[str], int]]
BondAttribute = Optional[Union[bool, str, int, float]]
AtomAttribute = Optional[Union[bool, str, int, float, ChiralSpec]]
Atom = Dict[str, AtomAttribute]
Bond = Dict[str, BondAttribute]
Chain = Tuple[List[Atom], List[Bond]]
Rnum = Tuple[int, Optional[float]]
Rnums = Dict[int, Rnum]
AtomRnums = List[Tuple[int, Optional[float]]]
NeighborSpec = Tuple[Optional[str], Optional[float]]
AtomPattern = Dict[NeighborSpec, Tuple[Vector, ...]]
AtomPatterns = Dict[Optional[str], List[Tuple[AtomPattern, float]]]
NeighborConstraints = Dict[int, Vector]
ChainPattern = Tuple[Tuple[Vector, Vector], Dict[int, Tuple[Vector, ...]]]
ConstraintsCandidates = Dict[
    Tuple[int],
    Tuple[
        List[List[NeighborConstraints]],
        List[float],
    ],
]
GraphCoordinates = Dict[int, Vector]