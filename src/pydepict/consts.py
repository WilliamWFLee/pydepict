#!/usr/bin/env python3

"""
pydepict.consts

Constants for parsing, depicting and rendering

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

from cmath import pi
from typing import Dict, List, Optional, Tuple, Union

from .models import Vector

# CUSTOM TYPES

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
AtomPattern = List[
    Tuple[
        Dict[NeighborSpec, Tuple[Vector, ...]],
        float,
    ]
]
NeighborConstraints = Dict[int, Vector]
ChainPattern = Tuple[Tuple[Vector, Vector], Dict[int, Tuple[Vector, ...]]]
ConstraintsCandidates = Dict[
    Tuple[int],
    Tuple[
        List[List[NeighborConstraints]],
        List[float],
    ],
]

# GENERAL CHEMISTRY DATA

HALOGENS = frozenset("F Cl Br I At".split())

# GEOMETRY

THIRTY_DEGREES = pi / 6

# PARSER ELEMENT SYMBOLS

WILDCARD = "*"

STANDARD_SYMBOLS = frozenset(
    (
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu "
        "Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs "
        "Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl "
        "Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh "
        "Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og Br"
    ).split()
)
AROMATIC_SYMBOLS = frozenset("b c n o s p se as".split())
ELEMENT_SYMBOLS = STANDARD_SYMBOLS | AROMATIC_SYMBOLS | {WILDCARD}

STANDARD_ORGANIC_SYMBOLS = frozenset("B C N O S P F Cl Br I".split())
AROMATIC_ORGANIC_SYMBOLS = frozenset("b c n o s p".split())
ORGANIC_SYMBOLS = STANDARD_ORGANIC_SYMBOLS | AROMATIC_ORGANIC_SYMBOLS | {WILDCARD}

ELEMENT_SYMBOL_FIRST_CHARS = frozenset(element[0] for element in ELEMENT_SYMBOLS)
ORGANIC_SYMBOL_FIRST_CHARS = frozenset(element[0] for element in ORGANIC_SYMBOLS)

# PARSER CHIRALITY

CHIRALITY_CODES = frozenset("TH AL SP TB OH".split())
CHIRALITY_CODES_FIRST_CHARS = frozenset(code[0] for code in CHIRALITY_CODES)
CHIRALITY_RANGES = {
    "TH": 2,
    "AL": 2,
    "SP": 3,
    "TB": 20,
    "OH": 30,
}

# OTHER PARSER SYMBOLS

CHARGE_SYMBOLS = frozenset({"-", "+"})
TERMINATORS = frozenset({" ", "\t", "\r", "\n"})
BOND_TO_ORDER: Dict[str, float] = {
    "-": 1,
    "=": 2,
    "#": 3,
    "$": 4,
    "/": 1,
    "\\": 1,
}

# PARSER SPECIFICATIONS

MIN_CHARGE_MAGNITUDE = 15
VALENCES: Dict[str, Optional[Tuple[int, ...]]] = {
    "B": (3,),
    "C": (4,),
    "N": (3, 5),
    "O": (2,),
    "P": (3, 5),
    "S": (2, 4, 6),
    "F": (1,),
    "Cl": (1,),
    "Br": (1,),
    "*": None,
}

# PARSER TEMPLATES

DEFAULT_ATOM = {
    "isotope": None,
    "element": "*",
    "hcount": 0,
    "charge": 0,
    "class": None,
    "aromatic": False,
}
DEFAULT_BOND = {"order": 1}

# DEPICTER VECTORS

RRR = Vector(1, 0)
RRU = Vector(1, 0).rotate(THIRTY_DEGREES)
RUU = Vector(1, 0).rotate(THIRTY_DEGREES * 2)
UUU = Vector(1, 0).rotate(THIRTY_DEGREES * 3)
LUU = Vector(1, 0).rotate(THIRTY_DEGREES * 4)
LLU = Vector(1, 0).rotate(THIRTY_DEGREES * 5)
LLL = Vector(1, 0).rotate(THIRTY_DEGREES * 6)
LLD = Vector(1, 0).rotate(THIRTY_DEGREES * 7)
LDD = Vector(1, 0).rotate(THIRTY_DEGREES * 8)
DDD = Vector(1, 0).rotate(THIRTY_DEGREES * 9)
RDD = Vector(1, 0).rotate(THIRTY_DEGREES * 10)
RRD = Vector(1, 0).rotate(THIRTY_DEGREES * 11)

# DEPICTER CONSTRAINTS

ATOM_PATTERNS: Dict[Optional[str], AtomPattern] = {
    "C": [
        (
            {
                (None, None): (LLL,),
            },
            1,
        ),
        (
            {
                (None, 1): (LLD,),
                (None, 2): (RRD,),
            },
            1,
        ),
        (
            {
                ("C", 1): (LLD,),
                (None, None): (RRD,),
            },
            1,
        ),
        (
            {
                (None, 1): (LLL,),
                (None, 3): (RRR,),
            },
            1,
        ),
        (
            {
                ("C", 2): (LLL, RRR),
            },
            1,
        ),
        (
            {
                (None, 1): (LLD, RRD),
            },
            1,
        ),
        (
            {
                (None, 1): (LUU, LDD),
                ("C", 2): (RRR,),
            },
            1,
        ),
        (
            {
                (None, 1): (LLD, UUU, RRD),
            },
            1,
        ),
        (
            {
                ("O", None): (UUU,),
                (None, 1): (LLD, RRD),
            },
            1,
        ),
        (
            {
                ("O", None): (LLD,),
                (None, 1): (UUU, RRD),
            },
            1,
        ),
        (
            {
                ("C", 1): (RRR, DDD, LLL, UUU),
            },
            1,
        ),
        (
            {
                ("C", 1): (RRR, RDD, LLL, RUU),
            },
            1,
        ),
        (
            {
                ("C", 1): (LLD,),
                (None, 1): (RRD,),
                ("X", 1): (LUU, RUU),
            },
            1,
        ),
        (
            {
                ("X", 1): (RRR, RDD, RUU),
                ("C", 1): (LLL,),
            },
            1,
        ),
        (
            {
                ("C", 1): (LLD, RRD),
                (None, 1): (LUU, RUU),
            },
            1,
        ),
        (
            {
                (None, 1): (UUU, RRR),
                ("C", 1): (DDD, LLL),
            },
            0.2,
        ),
        (
            {
                (None, 1): (UUU, DDD),
                ("C", 1): (LLL, RRR),
            },
            0.2,
        ),
    ],
    "N": [
        (
            {
                (None, 1): (LLD, RRD),
            },
            1,
        ),
        (
            {
                (None, 2): (LLD,),
                (None, 1): (RRD,),
            },
            1,
        ),
        (
            {
                (None, 2): (LLL,),
                (None, 1): (RRR,),
            },
            0.1,
        ),
    ],
    "O": [
        (
            {
                (None, 1): (LLD, RRD),
            },
            1,
        ),
        (
            {
                (None, 1): (LLL, RRR),
            },
            0.1,
        ),
    ],
    "P": [
        (
            {
                (None, 1): (LLD, RRD),
            },
            1,
        ),
        (
            {
                (None, 1): (LLL, RRR),
            },
            0.2,
        ),
        (
            {
                (None, 1): (LLL, RRR),
                ("O", 2): (UUU, DDD),
            },
            1,
        ),
        (
            {
                (None, 1): (RRR, DDD),
                ("O", 2): (LLL, UUU),
            },
            0.1,
        ),
    ],
    "S": [
        (
            {
                (None, 1): (LLD, RRD),
            },
            1,
        ),
        (
            {
                (None, 1): (LLL, RRR),
            },
            0.4,
        ),
        (
            {
                (None, 1): (LLL, RRR),
                ("O", 2): (UUU, DDD),
            },
            1,
        ),
        (
            {
                (None, 1): (RRR, DDD),
                ("O", 2): (LLL, UUU),
            },
            0.1,
        ),
    ],
    "Se": [
        (
            {
                (None, 1): (LLD, RRD),
            },
            1,
        ),
        (
            {
                (None, 1): (LLL, RRR),
            },
            0.5,
        ),
    ],
    "Si": [
        (
            {
                (None, 1): (LLD, RRD),
            },
            1,
        ),
        (
            {
                (None, 1): (LLL, RRR),
            },
            0.5,
        ),
    ],
    None: [
        (
            {
                (None, None): (LLL,),
            },
            1,
        ),
        (
            {
                (None, None): (LLL, RRR),
            },
            1,
        ),
        (
            {
                (None, 1): (LLD, RRD),
            },
            1,
        ),
        (
            {
                (None, None): (LLL, RUU, RDD),
            },
            1,
        ),
        (
            {
                (None, None): (UUU, RRR, DDD, LLL),
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
del meth, patterns, patterns_copy

CHAIN_PATTERN_UNITS: List[Tuple[ChainPattern, ChainPattern]] = [
    (
        (
            (LLD, RRD),
            {
                1: (UUU,),
                2: (LUU, RUU),
            },
        ),
        (
            (LLU, RRU),
            {
                1: (DDD,),
                2: (LDD, RDD),
            },
        ),
    ),
    (
        (
            (DDD, RRU),
            {
                1: (LLU,),
                2: (LLL, LUU),
            },
        ),
        (
            (LLD, UUU),
            {
                1: (RRD,),
                2: (RRR, RDD),
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

# OTHER DEPICTER CONSTANTS

CHAIN_ELEMENTS = frozenset("C N O S".split())
SAMPLE_SIZE = 100
EPSILON = 0.0001

# RENDERER WINDOW ATTRIBUTES

WINDOW_TITLE = "pydepict"

# RENDERER COLORS

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# RENDERER DIMENSIONS

MIN_DISPLAY_SIZE = (400, 300)
FRAME_MARGIN = 50
DISPLAY_BOND_LENGTH = 55
BOND_WIDTH = 6
TEXT_MARGIN = 2

# RENDERER FONT

FONT_FAMILY = "Arial"
FONT_SIZE = 20

# RENDERER PATHS
SCREENSHOTS_DIR = "screenshots"
