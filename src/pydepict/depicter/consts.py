#!/usr/bin/env python3

"""
pydepict.depicter.consts

Constants for the depicter.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details.
"""


from typing import List, Tuple

from ..models import Vector
from ..types import AtomPatterns, ChainPattern

# CONSTRAINTS

ATOM_PATTERNS: AtomPatterns = {
    "C": [
        (
            {
                (None, None): (Vector.LLL,),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.LLD,),
                (None, 2): (Vector.RRD,),
            },
            1,
        ),
        (
            {
                ("C", 1): (Vector.LLD,),
                (None, None): (Vector.RRD,),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.LLL,),
                (None, 3): (Vector.RRR,),
            },
            1,
        ),
        (
            {
                ("C", 2): (Vector.LLL, Vector.RRR),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.LLD, Vector.RRD),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.LUU, Vector.LDD),
                ("C", 2): (Vector.RRR,),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.LLD, Vector.UUU, Vector.RRD),
            },
            1,
        ),
        (
            {
                ("O", None): (Vector.UUU,),
                (None, 1): (Vector.LLD, Vector.RRD),
            },
            1,
        ),
        (
            {
                ("O", None): (Vector.LLD,),
                (None, 1): (Vector.UUU, Vector.RRD),
            },
            1,
        ),
        (
            {
                ("C", 1): (Vector.RRR, Vector.DDD, Vector.LLL, Vector.UUU),
            },
            1,
        ),
        (
            {
                ("C", 1): (Vector.RRR, Vector.RDD, Vector.LLL, Vector.RUU),
            },
            1,
        ),
        (
            {
                ("C", 1): (Vector.LLD,),
                (None, 1): (Vector.RRD,),
                ("X", 1): (Vector.LUU, Vector.RUU),
            },
            1,
        ),
        (
            {
                ("X", 1): (Vector.RRR, Vector.RDD, Vector.RUU),
                ("C", 1): (Vector.LLL,),
            },
            1,
        ),
        (
            {
                ("C", 1): (Vector.LLD, Vector.RRD),
                (None, 1): (Vector.LUU, Vector.RUU),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.UUU, Vector.RRR),
                ("C", 1): (Vector.DDD, Vector.LLL),
            },
            0.2,
        ),
        (
            {
                (None, 1): (Vector.UUU, Vector.DDD),
                ("C", 1): (Vector.LLL, Vector.RRR),
            },
            0.2,
        ),
    ],
    "N": [
        (
            {
                (None, 1): (Vector.LLD, Vector.RRD),
            },
            1,
        ),
        (
            {
                (None, 2): (Vector.LLD,),
                (None, 1): (Vector.RRD,),
            },
            1,
        ),
        (
            {
                (None, 2): (Vector.LLL,),
                (None, 1): (Vector.RRR,),
            },
            0.1,
        ),
    ],
    "O": [
        (
            {
                (None, 1): (Vector.LLD, Vector.RRD),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.LLL, Vector.RRR),
            },
            0.1,
        ),
    ],
    "P": [
        (
            {
                (None, 1): (Vector.LLD, Vector.RRD),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.LLL, Vector.RRR),
            },
            0.2,
        ),
        (
            {
                (None, 1): (Vector.LLL, Vector.RRR),
                ("O", 2): (Vector.UUU, Vector.DDD),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.RRR, Vector.DDD),
                ("O", 2): (Vector.LLL, Vector.UUU),
            },
            0.1,
        ),
    ],
    "S": [
        (
            {
                (None, 1): (Vector.LLD, Vector.RRD),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.LLL, Vector.RRR),
            },
            0.4,
        ),
        (
            {
                (None, 1): (Vector.LLL, Vector.RRR),
                ("O", 2): (Vector.UUU, Vector.DDD),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.RRR, Vector.DDD),
                ("O", 2): (Vector.LLL, Vector.UUU),
            },
            0.1,
        ),
    ],
    "Se": [
        (
            {
                (None, 1): (Vector.LLD, Vector.RRD),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.LLL, Vector.RRR),
            },
            0.5,
        ),
    ],
    "Si": [
        (
            {
                (None, 1): (Vector.LLD, Vector.RRD),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.LLL, Vector.RRR),
            },
            0.5,
        ),
    ],
    None: [
        (
            {
                (None, None): (Vector.LLL,),
            },
            1,
        ),
        (
            {
                (None, None): (Vector.LLL, Vector.RRR),
            },
            1,
        ),
        (
            {
                (None, 1): (Vector.LLD, Vector.RRD),
            },
            1,
        ),
        (
            {
                (None, None): (Vector.LLL, Vector.RUU, Vector.RDD),
            },
            1,
        ),
        (
            {
                (None, None): (Vector.UUU, Vector.RRR, Vector.DDD, Vector.LLL),
            },
            1,
        ),
    ],
}

# Calculates reflected atom constraints
for meth in (Vector.x_reflect, Vector.y_reflect):
    for patterns in ATOM_PATTERNS.values():
        patterns_copy = patterns.copy()
        patterns.extend(
            (
                {
                    atom: tuple(meth(v) for v in vectors)
                    for atom, vectors in pattern.items()
                },
                weight,
            )
            for pattern, weight in patterns_copy
        )

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
