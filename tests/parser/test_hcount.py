#!/usr/bin/env python3

"""
tests.parser.test_hydrogens

Tests the parsing of hydrogen counts
"""

import string

import pytest

from pydepict.parser import parse_hcount

from .utils import apply_parse_function

BRACKET_ATOM_TEMPLATE = "[*{}]"


@pytest.mark.parametrize("digit", string.digits)
def test_parse_hcount(digit: str):
    """
    Tests explicit hydrogen counts, e.g. [*H4].
    """
    result = apply_parse_function(parse_hcount, "H" + digit)
    assert result == int(digit)


def test_parse_implied_single_hcount():
    """
    Tests implied hydrogen count for 'H' symbol only, i.e. [*H] implies 1.
    """
    result = apply_parse_function(parse_hcount, "H")
    assert result == 1


def test_parse_implied_no_hcount():
    """
    Tests implied hydrogen count for no hydrogen count, i.e. [*] implies 0.
    """
    result = apply_parse_function(parse_hcount, "")
    assert result == 0
