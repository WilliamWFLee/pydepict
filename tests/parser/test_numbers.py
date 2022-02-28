#!/usr/bin/env python3

"""
tests.parser.test_numbers

Tests the parsing of charges
"""

import pytest
import pytest_mock

from pydepict.errors import ParserError
from pydepict.parser import parse_digit, parse_number

from .utils import apply_stream_parse_method, patch_parse_method


@pytest.fixture
def number(valid_number: int, mocker: pytest_mock.MockerFixture) -> str:
    num_as_str = str(valid_number)
    patch_parse_method(mocker, parse_digit, side_effect=num_as_str)
    return num_as_str


@pytest.fixture
def padded_number(valid_number: int, mocker: pytest_mock.MockerFixture) -> str:
    num_as_str = f"{valid_number:0>4}"
    patch_parse_method(mocker, parse_digit, side_effect=num_as_str)
    return num_as_str


def test_valid_number(number: str):
    """
    Tests parsing valid numbers of various orders of magnitude
    """
    result = apply_stream_parse_method(parse_number, number)
    assert result == int(number)


def test_valid_padded_number(padded_number: str):
    """
    Tests parsing valid numbers that have been padded with zero
    """
    result = apply_stream_parse_method(parse_number, padded_number)
    assert result == int(padded_number)


def test_no_number():
    """
    Tests parsing no number, with expected :class:`ParserError`
    """
    with pytest.raises(ParserError):
        apply_stream_parse_method(parse_number, "")
