#!/usr/bin/env python3

"""
tests.parser.test_digit

Tests the parsing of element symbols
"""

import string

import pytest

from pydepict.errors import ParserError
from pydepict.parser import Stream, parse_digit


@pytest.mark.parametrize("digit", string.digits)
def test_valid_digits(digit: str):
    """
    Tests valid digits
    """
    stream = Stream(digit)
    result = parse_digit(stream)
    assert result == digit


@pytest.mark.parametrize(
    "digit", string.punctuation + string.ascii_lowercase + string.ascii_uppercase
)
def test_invalid_digits(digit: str):
    """
    Tests invalid digits, using punctuation, and lowercase and uppercase letters
    """
    stream = Stream(digit)
    with pytest.raises(ParserError):
        parse_digit(stream)
