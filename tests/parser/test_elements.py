#!/usr/bin/env python3

"""
tests.parser.test_elements

Tests the parsing of element symbols
"""

import pytest

from pydepict.consts import ELEMENTS
from pydepict.errors import ParserError
from pydepict.parser import Stream, parse_element_symbol


@pytest.mark.parametrize("element", ELEMENTS)
def test_parse_valid_symbols(element: str):
    """
    Tests parsing a stream of a single element symbol
    """
    stream = Stream(element)
    result = parse_element_symbol(stream)
    assert result == element


@pytest.mark.parametrize(
    "element", [element.lower() for element in ELEMENTS if element.isalpha()]
)
def test_parse_lowercase_symbols(element: str):
    """
    Tests parsing lowercase element symbols, which should return :data:`None`
    """
    stream = Stream(element)
    with pytest.raises(ParserError):
        parse_element_symbol(stream)


@pytest.mark.parametrize("element", ("Fg", "Ak", "Of", "My", "Dj"))
def test_parse_nonexistent_symbols(element):
    """
    Tests parsing nonexistent element symbols, which should raise an error
    """
    stream = Stream(element)
    with pytest.raises(ParserError):
        parse_element_symbol(stream)
