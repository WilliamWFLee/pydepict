#!/usr/bin/env python3

"""
tests.parser.test_elements

Tests the parsing of element symbols
"""

import pytest

from pydepict.errors import ParserError

from .utils import apply_parse_method


def test_parse_valid_organic_symbols(valid_organic: str):
    """
    Tests parsing a stream of a single organic symbol
    """
    result = apply_parse_method("organic_symbol", valid_organic)
    assert result == valid_organic


def test_parse_invalid_organic_symbols(invalid_element: str):
    """
    Tests parsing non-existent element symbols, with expected :class:`ParserError`
    """
    with pytest.raises(ParserError):
        apply_parse_method("organic_symbol", invalid_element)


def test_parse_valid_element_not_organic(valid_element_not_organic: str):
    """
    Tests parsing valid element symbols, which cannot be used in an organic context
    """
    with pytest.raises(ParserError):
        apply_parse_method("organic_symbol", valid_element_not_organic)


def test_parse_no_organic_symbol():
    """
    Tests parsing no organic symbol, with expected :class:`ParserError`
    """
    with pytest.raises(ParserError):
        apply_parse_method("organic_symbol", "")
