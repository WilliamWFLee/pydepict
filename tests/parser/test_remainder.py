#!/usr/bin/env python3

"""
tests.parser.test_remainder

Tests the remainder of the SMILES string after parsing
"""

import random
import string

import pytest

from pydepict import parser
from pydepict.parser import Stream

STRING_LENGTH = 20


@pytest.fixture
def random_string() -> str:
    return "".join(random.choices(string.ascii_letters, k=20))


@pytest.fixture
def stream(random_string: str) -> Stream[str]:
    return Stream(random_string)


@pytest.mark.parametrize("offset", range(0, STRING_LENGTH + 1))
def test_remainder(stream: Stream[str], random_string: str, offset: int):
    for _ in range(offset):
        next(stream)
    result = parser.get_remainder(stream)
    assert result == random_string[offset:]
