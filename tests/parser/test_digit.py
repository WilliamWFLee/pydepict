#!/usr/bin/env python3

"""
tests.parser.test_digit

Tests the parsing of element symbols
"""

import pytest

from pydepict.errors import ParserError
from tests.parser.utils import apply_parse_method


def test_valid_digits(valid_digit: str):
    """
    Tests valid digits
    """
    result = apply_parse_method("digit", valid_digit)
    assert result == valid_digit


def test_invalid_digits(invalid_digit: str):
    """
    Tests invalid digits, using punctuation, and lowercase and uppercase letters
    """
    with pytest.raises(ParserError):
        apply_parse_method("digit", invalid_digit)
