#!/usr/bin/env python3

"""
tests.parser.test_bonds

Tests the parsing of bond symbols
"""

import pytest

from pydepict.consts import BOND_TO_ORDER
from pydepict.errors import ParserError
from pydepict.parser import parse_bond

from .utils import apply_stream_parse_method


def test_valid_bond(valid_bond: str):
    """
    Tests parsing valid bond symbols
    """
    result = apply_stream_parse_method(parse_bond, valid_bond)
    assert result["order"] == BOND_TO_ORDER[valid_bond]


def test_invalid_bond(invalid_bond: str):
    """
    Tests parsing invalid bond symbols
    """
    with pytest.raises(ParserError):
        apply_stream_parse_method(parse_bond, invalid_bond)
