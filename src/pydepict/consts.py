#!/usr/bin/env python3

"""
pydepict.consts

Constants such as element symbols, bond orders, other parsing symbols

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

from typing import Dict, List, Optional, Tuple, Union

# ELEMENT SYMBOLS

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

# CHIRALITY

CHIRALITY_CODES = frozenset("TH AL SP TB OH".split())
CHIRALITY_CODES_FIRST_CHARS = frozenset(code[0] for code in CHIRALITY_CODES)
CHIRALITY_RANGES = {
    "TH": 2,
    "AL": 2,
    "SP": 3,
    "TB": 20,
    "OH": 30,
}

# OTHER SYMBOLS

CHARGE_SYMBOLS = frozenset({"-", "+"})
TERMINATORS = frozenset({" ", "\t", "\r", "\n"})
BOND_TO_ORDER: Dict[str, float] = {
    "-": 1,
    "=": 2,
    "#": 3,
    "$": 4,
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

# RENDERER COLORS

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# TYPE CONSTANTS

ChiralSpec = Optional[Tuple[Optional[str], int]]
BondAttribute = Optional[Union[bool, str, int, float]]
AtomAttribute = Optional[Union[bool, str, int, float, ChiralSpec]]
Atom = Dict[str, AtomAttribute]
Bond = Dict[str, BondAttribute]
Chain = Tuple[List[Atom], List[Bond]]
Rnum = Tuple[int, Optional[float]]
Rnums = Dict[int, Rnum]
AtomRnums = List[Tuple[int, Optional[float]]]
