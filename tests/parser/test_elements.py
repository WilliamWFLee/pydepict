#!/usr/bin/env python3

"""
tests.parser.test_elements

Tests the parsing of element symbols
"""

import pytest

from pydepict.consts import CLOSE_BRACKET, ELEMENTS, OPEN_BRACKET
from pydepict.errors import ParserError
from pydepict.parser import Stream, parse, parse_atom, parse_element_symbol

BRACKET_ATOM_TEMPLATE = f"{OPEN_BRACKET}{{}}{CLOSE_BRACKET}"

NONEXISTENT_SYMBOLS = "Fg Ak Of My Dj".split()


@pytest.mark.parametrize("element", ELEMENTS)
def test_parse_valid_symbols(element: str):
    """
    Tests parsing a stream of a single element symbol
    """
    stream = Stream(element)
    result = parse_element_symbol(stream)
    assert result == element

    stream = Stream(BRACKET_ATOM_TEMPLATE.format(element))
    result = parse_atom(stream)
    assert result["element"] == element

    stream = Stream(BRACKET_ATOM_TEMPLATE.format(element))
    result = parse(stream)
    assert result.nodes[0]["element"] == element


@pytest.mark.parametrize(
    "element",
    [element.lower() for element in ELEMENTS if element.isalpha()]  # Lowercase symbols
    + NONEXISTENT_SYMBOLS,
)
def test_parse_invalid_symbols(element: str):
    """
    Tests parsing lowercase element symbols, which should return :data:`None`
    """
    stream = Stream(element)
    with pytest.raises(ParserError):
        parse_element_symbol(stream)

    stream = Stream(BRACKET_ATOM_TEMPLATE.format(element))
    with pytest.raises(ParserError):
        parse_atom(stream)

    stream = Stream(BRACKET_ATOM_TEMPLATE.format(element))
    with pytest.raises(ParserError):
        parse(stream)
