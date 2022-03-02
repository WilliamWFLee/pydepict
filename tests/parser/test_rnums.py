#!/usr/bin/env python3

"""
tests.parser.test_rnums

Tests the remainder of the SMILES string after parsing
"""

import pytest
import pytest_mock

from pydepict.errors import ParserError
from pydepict.parser import parse_digit, parse_rnum

from .utils import apply_stream_parse_method, patch_parse_method


@pytest.fixture
def valid_single_digit_rnum(mocker: pytest_mock.MockerFixture, valid_digit: str) -> str:
    patch_parse_method(mocker, parse_digit, return_value=valid_digit)
    return valid_digit


@pytest.fixture
def invalid_single_digit_rnum(
    mocker: pytest_mock.MockerFixture, invalid_digit: str
) -> str:
    patch_parse_method(mocker, parse_digit, side_effect=(invalid_digit,))
    return invalid_digit


@pytest.fixture
def valid_double_digit_rnum(
    mocker: pytest_mock.MockerFixture, valid_double_digits: str
) -> str:
    patch_parse_method(mocker, parse_digit, side_effect=valid_double_digits)
    return valid_double_digits


def test_valid_single_digit_rnums(valid_single_digit_rnum: str):
    result = apply_stream_parse_method(parse_rnum, valid_single_digit_rnum)
    assert result == int(valid_single_digit_rnum)


def test_valid_double_digit_rnums(valid_double_digit_rnum: str):
    result = apply_stream_parse_method(parse_rnum, "%" + valid_double_digit_rnum)
    assert result == int(valid_double_digit_rnum)


def test_invalid_single_digit_rnums(invalid_single_digit_rnum: str):
    with pytest.raises(ParserError):
        apply_stream_parse_method(parse_rnum, invalid_single_digit_rnum)
