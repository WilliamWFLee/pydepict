#!/usr/bin/env python3

"""
tests.parser.test_charges

Tests the parsing of charges
"""

import pytest
import pytest_mock

from pydepict.errors import ParserError
from tests.parser.utils import apply_parse_method, patch_parse_method


@pytest.fixture
def class_(valid_class: int, mocker: pytest_mock.MockerFixture) -> str:
    patch_parse_method(mocker, "number", valid_class)
    return f":{valid_class}"


@pytest.fixture
def padded_class(valid_class: int, mocker: pytest_mock.MockerFixture) -> str:
    patch_parse_method(mocker, "number", valid_class)
    return f":{valid_class:0>4}"


def test_valid_class(class_: str):
    """
    Tests parsing an atom class
    """
    result = apply_parse_method("class", class_)
    assert result == int(class_[1:])


def test_valid_padded_class(padded_class: str):
    """
    Tests parsing an 0-padded atom class
    """
    result = apply_parse_method("class", padded_class)
    assert result == int(padded_class[1:])


def test_no_class():
    """
    Tests parsing no atom class, with expected :class:`ParserError`
    """
    with pytest.raises(ParserError):
        apply_parse_method("class", "")


def test_no_class_number():
    """
    Tests parsing no atom class number, with expected :class:`ParserError`
    """
    with pytest.raises(ParserError):
        apply_parse_method("class", ":")
