#!/usr/bin/env python3

"""
tests.parser.test_atoms

Tests the parsing of atom strings
"""

from argparse import ArgumentError
from typing import Optional
from unittest.mock import DEFAULT

import pytest
import pytest_mock

from pydepict.consts import Atom, AtomAttribute
from pydepict.parser import Parser, Stream
from tests.parser.utils import apply_parse_method, patch_parse_method

SINGLE_ATOM_TEMPLATE = "[{isotope}{element}H{hcount}{charge:+}]"


@pytest.fixture(scope="module")
def isotope() -> int:
    return 1343


@pytest.fixture(scope="module")
def element() -> str:
    return "*"


@pytest.fixture(scope="module")
def hcount() -> int:
    return 4


@pytest.fixture(scope="module")
def charge() -> int:
    return 2


@pytest.fixture(scope="module")
def smiles(
    isotope: int,
    element: str,
    hcount: int,
    charge: int,
) -> Atom:
    attrs = {
        "isotope": isotope,
        "element": element,
        "hcount": hcount,
        "charge": charge,
    }

    return SINGLE_ATOM_TEMPLATE.format(**attrs)


@pytest.fixture(scope="module")
def stream(smiles: str) -> Stream:
    return Stream(smiles)


@pytest.fixture(scope="module")
def atom(
    isotope: int,
    element: str,
    hcount: int,
    charge: int,
    stream: str,
    module_mocker: pytest_mock.MockerFixture,
) -> Atom:
    def increment_stream_pos_by(value: int):
        """
        Returns a :class:`typing.Callable` that can be called
        to increment the stream position
        """

        def inner(*args, **kwargs):
            for _ in range(value):
                next(stream)
            return DEFAULT

        return inner

    def length(attr: str, value: AtomAttribute):
        if attr == "element_symbol":
            return len(value)
        if attr == "isotope":
            return len(str(value))
        if attr == "hcount":
            return len(str(value)) + 1
        if attr == "charge":
            return len(f"{value:+}")
        raise ArgumentError("attr", "is not a recognised element attribute")

    for attr, value in [
        ("isotope", isotope),
        ("element_symbol", element),
        ("hcount", hcount),
        ("charge", charge),
    ]:
        patch_parse_method(
            module_mocker, attr, value, increment_stream_pos_by(length(attr, value))
        )

    return apply_parse_method(Parser.parse_atom, stream)


def test_atom_isotope(atom: Atom, isotope: Optional[int]):
    assert atom["isotope"] == isotope


def test_atom_element(atom: Atom, element: str):
    assert atom["element"] == element


def test_atom_charge(atom: Atom, charge: int):
    assert atom["charge"] == charge


def test_atom_hcount(atom: Atom, hcount: int):
    assert atom["hcount"] == hcount
