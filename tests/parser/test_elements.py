#!/usr/bin/env python3

"""
tests.parser.test_elements

Tests the parsing of element symbols
"""

import pytest

from pydepict.errors import ParserError

from .utils import apply_parse_method

BRACKET_ATOM_TEMPLATE = "[{}]"


def test_parse_valid_symbols(valid_element: str):
    """
    Tests parsing a stream of a single element symbol
    """
    result = apply_parse_method("element_symbol", valid_element)
    assert result == valid_element


def test_parse_invalid_symbols(invalid_element: str):
    """
    Tests parsing lowercase element symbols, which should return :data:`None`
    """
    with pytest.raises(ParserError):
        apply_parse_method("element_symbol", invalid_element)


def test_parse_no_symbol():
    """
    Tests parsing no element symbol, with expected :class:`ParserError`
    """
    with pytest.raises(ParserError):
        apply_parse_method("element_symbol", "")
