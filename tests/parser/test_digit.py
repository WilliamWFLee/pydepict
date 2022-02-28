#!/usr/bin/env python3

"""
tests.parser.test_digit

Tests the parsing of element symbols
"""

import pytest

from pydepict.errors import ParserError
from pydepict.parser import parse_digit
from tests.parser.utils import apply_stream_parse_method


def test_valid_digits(valid_digit: str):
    """
    Tests valid digits
    """
    result = apply_stream_parse_method(parse_digit, valid_digit)
    assert result == valid_digit


def test_invalid_digits(invalid_digit: str):
    """
    Tests invalid digits, using punctuation, and lowercase and uppercase letters
    """
    with pytest.raises(ParserError):
        apply_stream_parse_method(parse_digit, invalid_digit)
