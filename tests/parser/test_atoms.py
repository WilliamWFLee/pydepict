#!/usr/bin/env python3

"""
tests.parser.test_atoms

Tests the parsing of atom strings
"""

from typing import Dict
from unittest.mock import DEFAULT

import pytest
import pytest_mock

import pydepict
from pydepict.consts import AtomAttribute
from pydepict.parser import Stream, parse_atom

SINGLE_ATOM_TEMPLATE = "[{element}H{hcount}{charge:+}]"


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
def smiles(element: str, hcount: int, charge: int) -> int:
    attrs = {
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
    element: str,
    hcount: int,
    charge: int,
    stream: str,
    module_mocker: pytest_mock.MockerFixture,
) -> Dict[str, AtomAttribute]:
    def increment_stream_pos_by(value: int):
        def inner(*args, **kwargs):
            for _ in range(value):
                next(stream)
            return DEFAULT

        return inner

    for attr, value, offset in [
        ("element_symbol", element, 0),
        ("hcount", hcount, 1),
        ("charge", charge, 1),
    ]:
        mock = module_mocker.patch.object(pydepict.parser, f"parse_{attr}")
        mock.return_value = value
        mock.side_effect = increment_stream_pos_by(len(str(value)) + offset)

    return parse_atom(stream)


def test_atom_element(atom: Dict[str, AtomAttribute], element: str):
    assert atom["element"] == element


def test_atom_charge(atom: Dict[str, AtomAttribute], charge: int):
    assert atom["charge"] == charge


def test_atom_hcount(atom: Dict[str, AtomAttribute], hcount: int):
    assert atom["hcount"] == hcount
