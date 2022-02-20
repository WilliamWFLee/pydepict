#!/usr/bin/env python3

"""
tests.parser.test_isotopes

Tests the parsing of isotope specifications
"""

import pytest
import pytest_mock

from pydepict.parser import Parser

from .utils import apply_parse_method, patch_parse_method


@pytest.fixture
def isotope(valid_isotope: int, module_mocker: pytest_mock.MockerFixture) -> int:
    patch_parse_method(module_mocker, "number", return_value=valid_isotope)
    return valid_isotope


def test_valid_isotope(isotope: int):
    """
    Tests parsing valid isotope specs
    """
    result = apply_parse_method(Parser.parse_isotope, str(isotope))
    assert result == isotope


def test_valid_padded_isotope(isotope: int):
    """
    Tests parsing valid isotope specs that have been padded with zeros
    """
    result = apply_parse_method(Parser.parse_isotope, f"{isotope:0>4}")
    assert result == isotope
