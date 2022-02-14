#!/usr/bin/env python3

"""
tests.parser.test_charges

Tests the parsing of charges
"""

import pytest

from pydepict.consts import CHARGE_SYMBOLS
from pydepict.errors import ParserWarning
from pydepict.parser import parse, parse_atom, parse_charge

from .utils import apply_parse_function

MIN_CHARGE_MAGNITUDE = 15

ATOM_TEMPLATE = "[*{:+}]"
SYM_ONLY_ATOM_TEMPLATE = "[*{}]"
CHARGE_TEMPLATE = "{:+}"


@pytest.mark.parametrize(
    "charge", range(-MIN_CHARGE_MAGNITUDE, MIN_CHARGE_MAGNITUDE + 1)
)
def test_valid_charge(charge: int):
    """
    Tests valid explicit charges, e.g. [*+2]
    """
    result = apply_parse_function(parse_charge, CHARGE_TEMPLATE.format(charge))
    assert result == charge

    result = apply_parse_function(parse_atom, ATOM_TEMPLATE.format(charge))
    assert result["charge"] == charge

    result = apply_parse_function(parse, ATOM_TEMPLATE.format(charge))
    assert result.nodes[0]["charge"] == charge


@pytest.mark.parametrize("charge_sym", CHARGE_SYMBOLS)
def test_implied_charge_magnitude(charge_sym: str):
    """
    Tests the implied magnitude for a charge symbol, i.e. [*+] is interpreted as +1
    """
    charge = int(charge_sym + "1")

    result = apply_parse_function(parse_charge, charge_sym)
    assert result == charge

    result = apply_parse_function(parse_atom, SYM_ONLY_ATOM_TEMPLATE.format(charge_sym))
    assert result["charge"] == charge

    result = apply_parse_function(parse, SYM_ONLY_ATOM_TEMPLATE.format(charge_sym))
    assert result.nodes[0]["charge"] == charge


@pytest.mark.parametrize("charge_sym", CHARGE_SYMBOLS)
def test_double_charge_symbols(charge_sym: str):
    """
    Tests the double charge symbols, i.e. [*--] is interpreted as -2

    Should also raise deprecation warning.
    """
    charge = int(charge_sym + "2")

    with pytest.warns(ParserWarning):
        result = apply_parse_function(parse_charge, 2 * charge_sym)
    assert result == charge

    with pytest.warns(ParserWarning):
        result = apply_parse_function(
            parse_atom, SYM_ONLY_ATOM_TEMPLATE.format(2 * charge_sym)
        )
    assert result["charge"] == charge

    with pytest.warns(ParserWarning):
        result = apply_parse_function(
            parse, SYM_ONLY_ATOM_TEMPLATE.format(2 * charge_sym)
        )
    assert result.nodes[0]["charge"] == charge


def test_implied_no_charge():
    """
    Tests no charge specified implies no charge interpreted.
    """

    result = apply_parse_function(parse_charge, "")
    assert result == 0

    result = apply_parse_function(parse_atom, SYM_ONLY_ATOM_TEMPLATE.format(""))
    assert result["charge"] == 0

    result = apply_parse_function(parse, SYM_ONLY_ATOM_TEMPLATE.format(""))
    assert result.nodes[0]["charge"] == 0
