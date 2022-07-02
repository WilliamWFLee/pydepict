#!/usr/bin/env python3

"""
pydepict.depicter.consts

Constants for the depicter.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details.
"""


from typing import List, Tuple

from ..models import Vector
from ..types import AtomPatternsDict, ChainPattern
from .models import AtomPattern

# CONSTRAINTS

atom_patterns: AtomPatternsDict = {
    "C": [
        (
            {
                ((None, None), Vector.LLL),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.LLD),
                ((None, 2), Vector.RRD),
            },
            1,
        ),
        (
            {
                (("C", 1), Vector.LLD),
                ((None, None), Vector.RRD),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.LLL),
                ((None, 3), Vector.RRR),
            },
            1,
        ),
        (
            {
                (("C", 2), Vector.LLL),
                (("C", 2), Vector.RRR),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.LLD),
                ((None, 1), Vector.RRD),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.LUU),
                ((None, 1), Vector.LDD),
                (("C", 2), Vector.RRR),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.LLD),
                ((None, 1), Vector.UUU),
                ((None, 1), Vector.RRD),
            },
            1,
        ),
        (
            {
                (("O", None), Vector.UUU),
                ((None, 1), Vector.LLD),
                ((None, 1), Vector.RRD),
            },
            1,
        ),
        (
            {
                (("O", None), Vector.LLD),
                ((None, 1), Vector.UUU),
                ((None, 1), Vector.RRD),
            },
            1,
        ),
        (
            {
                (("C", 1), Vector.RRR),
                (("C", 1), Vector.DDD),
                (("C", 1), Vector.LLL),
                (("C", 1), Vector.UUU),
            },
            1,
        ),
        (
            {
                (("C", 1), Vector.RRR),
                (("C", 1), Vector.RDD),
                (("C", 1), Vector.LLL),
                (("C", 1), Vector.RUU),
            },
            1,
        ),
        (
            {
                (("C", 1), Vector.LLD),
                ((None, 1), Vector.RRD),
                (("X", 1), Vector.LUU),
                (("X", 1), Vector.RUU),
            },
            1,
        ),
        (
            {
                (("X", 1), Vector.RRR),
                (("X", 1), Vector.RDD),
                (("X", 1), Vector.RUU),
                (("C", 1), Vector.LLL),
            },
            1,
        ),
        (
            {
                (("C", 1), Vector.LLD),
                (("C", 1), Vector.RRD),
                ((None, 1), Vector.LUU),
                ((None, 1), Vector.RUU),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.UUU),
                ((None, 1), Vector.RRR),
                (("C", 1), Vector.DDD),
                (("C", 1), Vector.LLL),
            },
            0.2,
        ),
        (
            {
                ((None, 1), Vector.UUU),
                ((None, 1), Vector.DDD),
                (("C", 1), Vector.LLL),
                (("C", 1), Vector.RRR),
            },
            0.2,
        ),
    ],
    "N": [
        (
            {
                ((None, 1), Vector.LLD),
                ((None, 1), Vector.RRD),
            },
            1,
        ),
        (
            {
                ((None, 2), Vector.LLD),
                ((None, 1), Vector.RRD),
            },
            1,
        ),
        (
            {
                ((None, 2), Vector.LLL),
                ((None, 1), Vector.RRR),
            },
            0.1,
        ),
    ],
    "O": [
        (
            {
                ((None, 1), Vector.LLD),
                ((None, 1), Vector.RRD),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.LLL),
                ((None, 1), Vector.RRR),
            },
            0.1,
        ),
    ],
    "P": [
        (
            {
                ((None, 1), Vector.LLD),
                ((None, 1), Vector.RRD),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.LLL),
                ((None, 1), Vector.RRR),
            },
            0.2,
        ),
        (
            {
                ((None, 1), Vector.LLL),
                ((None, 1), Vector.RRR),
                (("O", 2), Vector.UUU),
                (("O", 2), Vector.DDD),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.RRR),
                ((None, 1), Vector.DDD),
                (("O", 2), Vector.LLL),
                (("O", 2), Vector.UUU),
            },
            0.1,
        ),
    ],
    "S": [
        (
            {
                ((None, 1), Vector.LLD),
                ((None, 1), Vector.RRD),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.LLL),
                ((None, 1), Vector.RRR),
            },
            0.4,
        ),
        (
            {
                ((None, 1), Vector.LLL),
                ((None, 1), Vector.RRR),
                (("O", 2), Vector.UUU),
                (("O", 2), Vector.DDD),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.RRR),
                ((None, 1), Vector.DDD),
                (("O", 2), Vector.LLL),
                (("O", 2), Vector.UUU),
            },
            0.1,
        ),
    ],
    "Se": [
        (
            {
                ((None, 1), Vector.LLD),
                ((None, 1), Vector.RRD),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.LLL),
                ((None, 1), Vector.RRR),
            },
            0.5,
        ),
    ],
    "Si": [
        (
            {
                ((None, 1), Vector.LLD),
                ((None, 1), Vector.RRD),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.LLL),
                ((None, 1), Vector.RRR),
            },
            0.5,
        ),
    ],
    None: [
        (
            {
                ((None, None), Vector.LLL),
            },
            1,
        ),
        (
            {
                ((None, None), Vector.LLL),
                ((None, None), Vector.RRR),
            },
            1,
        ),
        (
            {
                ((None, 1), Vector.LLD),
                ((None, 1), Vector.RRD),
            },
            1,
        ),
        (
            {
                ((None, None), Vector.LLL),
                ((None, None), Vector.RUU),
                ((None, None), Vector.RDD),
            },
            1,
        ),
        (
            {
                ((None, None), Vector.UUU),
                ((None, None), Vector.RRR),
                ((None, None), Vector.DDD),
                ((None, None), Vector.LLL),
            },
            1,
        ),
    ],
}

# Calculates reflected atom constraints
for meth in (Vector.x_reflect, Vector.y_reflect):
    for patterns in atom_patterns.values():
        patterns_copy = patterns.copy()
        patterns.extend(
            (
                {(neighbor_spec, meth(vector)) for neighbor_spec, vector in pattern},
                weight,
            )
            for pattern, weight in patterns_copy
        )

# Construct atom pattern instances

ATOM_PATTERNS = {
    center: [
        AtomPattern(center, neighbor_pattern, weight)
        for neighbor_pattern, weight in patterns
    ]
    for center, patterns in atom_patterns.items()
}

del atom_patterns

CHAIN_PATTERN_UNITS: List[Tuple[ChainPattern, ChainPattern]] = [
    (
        (
            (Vector.LLD, Vector.RRD),
            {
                1: (Vector.UUU,),
                2: (Vector.LUU, Vector.RUU),
            },
        ),
        (
            (Vector.LLU, Vector.RRU),
            {
                1: (Vector.DDD,),
                2: (Vector.LDD, Vector.RDD),
            },
        ),
    ),
    (
        (
            (Vector.DDD, Vector.RRU),
            {
                1: (Vector.LLU,),
                2: (Vector.LLL, Vector.LUU),
            },
        ),
        (
            (Vector.LLD, Vector.UUU),
            {
                1: (Vector.RRD,),
                2: (Vector.RRR, Vector.RDD),
            },
        ),
    ),
]

# Combine reflections and order swapping to produce other patterns

for pattern in CHAIN_PATTERN_UNITS.copy():
    first, second = pattern
    swapped = (second, first)
    if swapped not in CHAIN_PATTERN_UNITS:
        CHAIN_PATTERN_UNITS.append(swapped)
for meth in (Vector.x_reflect, Vector.y_reflect):
    for pattern in CHAIN_PATTERN_UNITS.copy():
        new_pattern = tuple(
            (
                (meth(prev_vector), meth(next_vector)),
                {
                    num_subs: tuple(meth(v) for v in vectors)
                    for num_subs, vectors in sub_constraints.items()
                },
            )
            for (prev_vector, next_vector), sub_constraints in pattern
        )
        if new_pattern not in CHAIN_PATTERN_UNITS:
            CHAIN_PATTERN_UNITS.append(new_pattern)

# OTHER CONSTANTS

CHAIN_ELEMENTS = frozenset("C N O S".split())
DEPICTION_SAMPLE_SIZE = 100
EPSILON = 0.0001
