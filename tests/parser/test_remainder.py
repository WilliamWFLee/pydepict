#!/usr/bin/env python3

"""
tests.parser.test_remainder

Tests the remainder of the SMILES string after parsing
"""

import random
import string

import pytest

from pydepict.parser import Parser

STRING_LENGTH = 20


@pytest.fixture
def random_string() -> str:
    return "".join(random.choices(string.ascii_letters, k=20))


@pytest.fixture
def parser(random_string: str) -> str:
    parser = Parser(random_string)
    parser._setup_parse()
    return parser


@pytest.mark.parametrize("offset", range(0, STRING_LENGTH + 1))
def test_remainder(parser: Parser, random_string: str, offset: int):
    for _ in range(offset):
        next(parser._stream)
    result = parser.get_remainder()
    assert result == random_string[offset:]
