#!/usr/bin/env python3

"""
tests.parser.test_charges

Tests the parsing of charges
"""

import pytest
import pytest_mock

from pydepict.errors import ParserWarning
from pydepict.parser import Parser

from .utils import apply_parse_method, patch_parse_method

ATOM_TEMPLATE = "[*{:+}]"
SYM_ONLY_ATOM_TEMPLATE = "[*{}]"
CHARGE_TEMPLATE = "{:+}"


@pytest.fixture
def charge(valid_charge: int, mocker: pytest_mock.MockerFixture) -> int:
    patch_parse_method(mocker, "digit", side_effect=str(abs(valid_charge)).split())
    return valid_charge


def test_valid_charge(charge: int):
    """
    Tests valid explicit charges, e.g. [*+2]
    """
    result = apply_parse_method(Parser.parse_charge, CHARGE_TEMPLATE.format(charge))
    assert result == charge


def test_implied_charge_magnitude(valid_symbol_charge: str):
    """
    Tests the implied magnitude for a charge symbol, i.e. [*+] is interpreted as +1
    """
    result = apply_parse_method(Parser.parse_charge, valid_symbol_charge)
    assert result == int(f"{valid_symbol_charge}1")


def test_double_charge_symbols(valid_double_symbol_charge: str):
    """
    Tests the double charge symbols, i.e. [*--] is interpreted as -2

    Should also raise deprecation warning.
    """
    with pytest.warns(ParserWarning):
        result = apply_parse_method(Parser.parse_charge, valid_double_symbol_charge)
    assert result == int(f"{valid_double_symbol_charge[0]}2")


def test_implied_no_charge():
    """
    Tests no charge specified implies no charge interpreted.
    """
    result = apply_parse_method(Parser.parse_charge, "")
    assert result == 0
