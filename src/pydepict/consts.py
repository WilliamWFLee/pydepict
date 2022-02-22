#!/usr/bin/env python3

"""
pydepict.consts

Constants such as element symbols, bond orders, other parsing symbols
"""

from typing import Dict, Optional, Union

# ELEMENT SYMBOLS

ELEMENT_SYMBOLS = (
    "* H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At "
    "Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn "
    "Nh Fl Mc Lv Ts Og"
).split()
ORGANIC_SYMBOLS = "B C N O S P F Cl Br I b c n o s p".split()

ELEMENT_SYMBOL_FIRST_CHARS = set(element[0] for element in ELEMENT_SYMBOLS)
ORGANIC_SYMBOL_FIRST_CHARS = set(element[0] for element in ORGANIC_SYMBOLS)

# OTHER SYMBOLS

CHARGE_SYMBOLS = ("-", "+")
TERMINATORS = (" ", "\t", "\r", "\n")
BOND_TO_ORDER: Dict[str, float] = {
    "-": 1,
    "=": 2,
    "#": 3,
    "$": 4,
}

# PARSER SPECIFICATIONS

MIN_CHARGE_MAGNITUDE = 15

# TYPE CONSTANTS

AtomAttribute = Union[bool, str, int, float]
Atom = Dict[str, Optional[AtomAttribute]]
