#!/usr/bin/env python3

"""
tests.parser.test_terminators

Tests the parsing of bond symbols
"""

from .utils import apply_parse_method


def test_valid_terminator(valid_terminator: str):
    apply_parse_method("terminator", valid_terminator)


def test_end_of_string():
    apply_parse_method("terminator", "")
