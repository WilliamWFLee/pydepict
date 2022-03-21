#!/usr/bin/env python3

"""
tests.parser.test_atoms

Tests the parsing of atom strings
"""

from typing import Optional
from unittest.mock import DEFAULT

import pytest
import pytest_mock

from pydepict import parser
from pydepict.consts import Atom, AtomAttribute
from pydepict.parser import Stream, parse_atom
from tests.parser.utils import apply_stream_parse_method, patch_parse_method

SINGLE_ATOM_TEMPLATE = "[{isotope}{element}H{hcount}{charge:+}:{class}]"


@pytest.fixture(scope="module", params=(0, 43, 684))
def isotope(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(scope="module", params=("Tm", "In", "b"))
def element(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module", params=(4, 6, 1))
def hcount(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(scope="module", params=(-12, 0, 6))
def charge(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(scope="module", params=(3, 15, 2352))
def class_(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(scope="module")
def smiles(
    isotope: int,
    element: str,
    hcount: int,
    charge: int,
    class_: int,
) -> Atom:
    attrs = {
        "isotope": isotope,
        "element": element,
        "hcount": hcount,
        "charge": charge,
        "class": class_,
    }

    return SINGLE_ATOM_TEMPLATE.format(**attrs)


@pytest.fixture
def stream(smiles: str) -> Stream:
    return Stream(smiles)


@pytest.fixture
def atom(
    isotope: int,
    element: str,
    hcount: int,
    charge: int,
    class_: int,
    stream: str,
    mocker: pytest_mock.MockerFixture,
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
        if attr in ("hcount", "class"):
            return len(str(value)) + 1
        if attr == "charge":
            return len(f"{value:+}")
        raise ValueError("attr", "is not a recognised element attribute")

    for attr, value in [
        ("isotope", isotope),
        ("element_symbol", element),
        ("hcount", hcount),
        ("charge", charge),
        ("class", class_),
    ]:
        patch_parse_method(
            mocker,
            getattr(parser, f"parse_{attr}"),
            value,
            increment_stream_pos_by(length(attr, value)),
        )

    return apply_stream_parse_method(parse_atom, stream)


def test_atom_isotope(atom: Atom, isotope: Optional[int]):
    assert atom["isotope"] == isotope


def test_atom_element(atom: Atom, element: str):
    assert atom["element"].title() == element.title()


def test_atom_aromaticity(atom: Atom, element: str):
    assert atom["aromatic"] == (True if element.islower() else False)


def test_atom_charge(atom: Atom, charge: int):
    assert atom["charge"] == charge


def test_atom_hcount(atom: Atom, hcount: int):
    assert atom["hcount"] == hcount


def test_atom_class(atom: Atom, class_: int):
    assert atom["class"] == class_
