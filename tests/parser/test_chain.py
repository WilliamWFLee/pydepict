#!/usr/bin/env python3

"""
tests.parser.test_chain

Tests the parsing of chains
"""

import pytest
import pytest_mock

from pydepict.consts import Chain
from pydepict.errors import ParserError
from pydepict.parser import Parser, _new_atom, _new_bond

from .utils import patch_parse_method


@pytest.fixture(params=[True, False])
def prev_aromatic(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(
    params=[
        (  # Explicit standard atoms and bonds
            [
                {"element": "C"},
                {"element": "C"},
                {"element": "N"},
                {"element": "C"},
                {"element": "C"},
                {"element": "C"},
            ],
            [
                {"order": 1},
                {"order": 1},
                {"order": 1},
                {"order": 1},
                {"order": 2},
                {"order": 1},
            ],
        ),
        (  # Implicit single bonds
            [
                {"element": "C"},
                {"element": "C"},
                {"element": "N"},
                {"element": "C"},
                {"element": "C"},
                {"element": "C"},
            ],
            [
                None,
                {"order": 1},
                None,
                None,
                {"order": 2},
                {"order": 1},
            ],
        ),
        (  # Aromatic atoms
            [
                {"element": "C", "aromatic": True},
                {"element": "C"},
                {"element": "N", "aromatic": True},
                {"element": "C"},
                {"element": "C"},
                {"element": "C"},
            ],
            [
                None,
                {"order": 1},
                None,
                None,
                {"order": 2},
                {"order": 1},
            ],
        ),
        (  # Aromatic atoms
            [
                {"element": "C", "aromatic": True},
                {"element": "C"},
                {"element": "C", "aromatic": True},
                {"element": "C"},
                {"element": "C"},
                {"element": "C"},
            ],
            [
                None,
                {"order": 1.5},
                None,
                None,
                {"order": 2},
                {"order": 1},
            ],
        ),
    ],
)
def chain(
    request: pytest.FixtureRequest,
    mocker: pytest_mock.MockerFixture,
    prev_aromatic: bool,
) -> Chain:
    atoms, bonds = request.param
    atoms = [_new_atom(**atom) for atom in atoms]

    patch_parse_method(mocker, "atom", side_effect=atoms)
    patch_parse_method(
        mocker,
        "bond",
        side_effect=[ParserError("", 0) if bond is None else bond for bond in bonds],
    )

    # Store aromatic attributes for each atom
    atom_aromatics = [atom["aromatic"] for atom in atoms]
    # Prepand aromatic attribute for previous atom
    atom_aromatics.insert(0, prev_aromatic)

    new_bonds = []
    # Bond i joins atoms (i-1) and i
    for left_aromatic, right_aromatic, bond in zip(
        atom_aromatics[:-1], atom_aromatics[1:], bonds
    ):
        if bond is None:
            if left_aromatic and right_aromatic:
                new_bonds.append(_new_bond(order=1.5))
            else:
                new_bonds.append(_new_bond(order=1))
        else:
            new_bonds.append(bond)

    return atoms, new_bonds


@pytest.fixture
def parser() -> Parser:
    parser = Parser("")
    parser._setup_parse()
    return parser


def test_valid_chain(parser: Parser, chain: Chain, prev_aromatic: bool):
    atoms, bonds = parser.parse_chain(prev_aromatic)
    assert atoms == chain[0]
    assert bonds == chain[1]
