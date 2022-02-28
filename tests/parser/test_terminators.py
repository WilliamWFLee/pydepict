#!/usr/bin/env python3

"""
tests.parser.test_terminators

Tests the parsing of bond symbols
"""

from pydepict.parser import parse_terminator

from .utils import apply_stream_parse_method


def test_valid_terminator(valid_terminator: str):
    apply_stream_parse_method(parse_terminator, valid_terminator)


def test_end_of_string():
    apply_stream_parse_method(parse_terminator, "")
