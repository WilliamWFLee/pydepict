#!/usr/bin/env python3

"""
tests.parser.test_hydrogens

Tests the parsing of hydrogen counts
"""

import pytest
import pytest_mock
from pydepict.parser import Parser

from .utils import apply_parse_method, patch_parse_method

BRACKET_ATOM_TEMPLATE = "[*{}]"


@pytest.fixture
def hcount(valid_hcount: str, mocker: pytest_mock.MockerFixture) -> str:
    patch_parse_method(mocker, "digit", int(valid_hcount))
    return valid_hcount


def test_parse_hcount(valid_hcount: str):
    """
    Tests explicit hydrogen counts, e.g. [*H4].
    """
    result = apply_parse_method(Parser.parse_hcount, "H" + valid_hcount)
    assert result == int(valid_hcount)


def test_parse_implied_single_hcount():
    """
    Tests implied hydrogen count for 'H' symbol only, i.e. [*H] implies 1.
    """
    result = apply_parse_method(Parser.parse_hcount, "H")
    assert result == 1


def test_parse_implied_no_hcount():
    """
    Tests implied hydrogen count for no hydrogen count, i.e. [*] implies 0.
    """
    result = apply_parse_method(Parser.parse_hcount, "")
    assert result == 0
