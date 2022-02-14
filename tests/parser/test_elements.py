#!/usr/bin/env python3

"""
tests.parser.test_elements

Tests the parsing of element symbols
"""

import pytest

from pydepict.consts import ELEMENTS
from pydepict.errors import ParserError
from pydepict.parser import parse, parse_atom, parse_element_symbol

from .utils import apply_parse_function

BRACKET_ATOM_TEMPLATE = "[{}]"

NONEXISTENT_SYMBOLS = "Fg Ak Of My Dj".split()


@pytest.mark.parametrize("symbol", ELEMENTS)
def test_parse_valid_symbols(symbol: str):
    """
    Tests parsing a stream of a single element symbol
    """
    result = apply_parse_function(parse_element_symbol, symbol)
    assert result == symbol

    result = apply_parse_function(parse_atom, BRACKET_ATOM_TEMPLATE.format(symbol))
    assert result["element"] == symbol

    result = apply_parse_function(parse, BRACKET_ATOM_TEMPLATE.format(symbol))
    assert result.nodes[0]["element"] == symbol


@pytest.mark.parametrize(
    "symbol",
    [element.lower() for element in ELEMENTS if element.isalpha()]  # Lowercase symbols
    + NONEXISTENT_SYMBOLS,
)
def test_parse_invalid_symbols(symbol: str):
    """
    Tests parsing lowercase element symbols, which should return :data:`None`
    """
    with pytest.raises(ParserError):
        apply_parse_function(parse_atom, symbol)

    with pytest.raises(ParserError):
        apply_parse_function(parse_atom, BRACKET_ATOM_TEMPLATE.format(symbol))

    with pytest.raises(ParserError):
        apply_parse_function(parse, BRACKET_ATOM_TEMPLATE.format(symbol))
