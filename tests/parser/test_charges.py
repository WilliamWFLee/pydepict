#!/usr/bin/env python3

"""
tests.parser.test_charges

Tests the parsing of charges
"""

import pytest
import pytest_mock

from pydepict.errors import ParserError, ParserWarning
from pydepict.models import Stream
from pydepict.parser import parse_charge, parse_digit

from .utils import apply_stream_parse_method, patch_parse_method

ATOM_TEMPLATE = "[*{:+}]"
SYM_ONLY_ATOM_TEMPLATE = "[*{}]"
CHARGE_TEMPLATE = "{:+}"


@pytest.fixture
def charge_stream(valid_charge: int) -> Stream[str]:
    return Stream(CHARGE_TEMPLATE.format(valid_charge))


@pytest.fixture
def charge(
    valid_charge: int, charge_stream: Stream[str], mocker: pytest_mock.MockerFixture
) -> int:
    def side_effect(arg):
        return next(charge_stream)

    patch_parse_method(mocker, parse_digit, side_effect=side_effect)
    return valid_charge


def test_valid_charge(charge_stream: int, charge: int):
    """
    Tests valid explicit charges, e.g. [*+2]
    """
    result = apply_stream_parse_method(parse_charge, charge_stream)
    assert result == charge


def test_implied_charge_magnitude(valid_symbol_charge: str):
    """
    Tests the implied magnitude for a charge symbol, i.e. [*+] is interpreted as +1
    """
    result = apply_stream_parse_method(parse_charge, valid_symbol_charge)
    assert result == int(f"{valid_symbol_charge}1")


def test_double_charge_symbols(valid_double_symbol_charge: str):
    """
    Tests the double charge symbols, i.e. [*--] is interpreted as -2

    Should also raise deprecation warning.
    """
    with pytest.warns(ParserWarning):
        result = apply_stream_parse_method(parse_charge, valid_double_symbol_charge)
    assert result == int(f"{valid_double_symbol_charge[0]}2")


def test_no_charge():
    """
    Tests no charge specified, with expected :class:`ParserError`
    """
    with pytest.raises(ParserError):
        apply_stream_parse_method(parse_charge, "")
