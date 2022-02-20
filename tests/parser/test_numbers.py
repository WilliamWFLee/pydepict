#!/usr/bin/env python3

"""
tests.parser.test_numbers

Tests the parsing of charges
"""

from pydepict.parser import Parser

from .utils import apply_parse_method


def test_valid_number(valid_number: int):
    """
    Tests parsing valid numbers of various orders of magnitude
    """
    result = apply_parse_method(Parser.parse_number, str(valid_number))
    assert result == valid_number


def test_valid_padded_number(valid_number: int):
    """
    Tests parsing valid numbers that have been padded with zero
    """
    result = apply_parse_method(Parser.parse_number, f"{valid_number:0>4}")
    assert result == valid_number
