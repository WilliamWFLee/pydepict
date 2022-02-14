#!/usr/bin/env python3

"""
tests.parser.test_digit

Tests the parsing of element symbols
"""

import string

import pytest

from pydepict.errors import ParserError
from pydepict.parser import parse_digit
from tests.parser.utils import apply_parse_function


@pytest.mark.parametrize("digit", string.digits)
def test_valid_digits(digit: str):
    """
    Tests valid digits
    """
    result = apply_parse_function(parse_digit, digit)
    assert result == digit


@pytest.mark.parametrize(
    "digit", string.punctuation + string.ascii_lowercase + string.ascii_uppercase
)
def test_invalid_digits(digit: str):
    """
    Tests invalid digits, using punctuation, and lowercase and uppercase letters
    """
    with pytest.raises(ParserError):
        apply_parse_function(parse_digit, digit)
