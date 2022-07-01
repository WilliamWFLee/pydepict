#!/usr/bin/env python3

"""
pydepict.consts

General constants.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details.
"""

from math import pi

# GENERAL CHEMISTRY DATA

HALOGENS = frozenset("F Cl Br I At".split())

# GEOMETRY

THIRTY_DEGS_IN_RADS = pi / 6

# VECTORS

VECTOR_NAMES = tuple("RRR RRU RUU UUU LUU LLU LLL LLD LDD DDD RDD RRD".split())
