#!/usr/bin/env python3

"""
tests.parser.test_hydrogens

Tests the parsing of hydrogen counts
"""

import string

import pytest

from pydepict.consts import HYDROGEN
from pydepict.parser import parse, parse_atom, parse_hcount

from .utils import apply_parse_function

BRACKET_ATOM_TEMPLATE = "[*{}]"


@pytest.mark.parametrize("digit", string.digits)
def test_parse_hcount(digit: str):
    result = apply_parse_function(parse_hcount, HYDROGEN + digit)
    assert result == int(digit)

    result = apply_parse_function(
        parse_atom, BRACKET_ATOM_TEMPLATE.format(HYDROGEN + digit)
    )
    assert result["hcount"] == int(digit)

    result = apply_parse_function(parse, BRACKET_ATOM_TEMPLATE.format(HYDROGEN + digit))
    assert result.nodes[0]["hcount"] == int(digit)


def test_parse_implied_single_hcount():
    result = apply_parse_function(parse_hcount, HYDROGEN)
    assert result == 1

    result = apply_parse_function(
        parse_atom, BRACKET_ATOM_TEMPLATE.format(HYDROGEN + "1")
    )
    assert result["hcount"] == 1

    result = apply_parse_function(parse, BRACKET_ATOM_TEMPLATE.format(HYDROGEN + "1"))
    assert result.nodes[0]["hcount"] == 1


def test_parse_implied_no_hcount():
    result = apply_parse_function(parse_hcount, "")
    assert result == 0

    result = apply_parse_function(
        parse_atom, BRACKET_ATOM_TEMPLATE.format(HYDROGEN + "0")
    )
    assert result["hcount"] == 0

    result = apply_parse_function(parse, BRACKET_ATOM_TEMPLATE.format(HYDROGEN + "0"))
    assert result.nodes[0]["hcount"] == 0
