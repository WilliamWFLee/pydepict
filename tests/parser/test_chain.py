#!/usr/bin/env python3

"""
tests.parser.test_chain

Tests the parsing of chains
"""

import pytest
import pytest_mock

from pydepict.errors import ParserError
from pydepict.parser import new_atom, new_bond, parse_atom, parse_bond, parse_chain
from pydepict.types import ParserChain

from .utils import apply_stream_parse_method, patch_parse_method


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
                {"element": "C", "aromatic": True},
                {"element": "C"},
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
        ),  #
    ],
)
def chain(
    request: pytest.FixtureRequest,
    mocker: pytest_mock.MockerFixture,
    prev_aromatic: bool,
) -> ParserChain:
    atoms, bonds = request.param
    atoms = [new_atom(**atom) for atom in atoms]

    patch_parse_method(mocker, parse_atom, side_effect=atoms)
    patch_parse_method(
        mocker,
        parse_bond,
        side_effect=[ParserError("") if bond is None else bond for bond in bonds],
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
                new_bonds.append(new_bond(order=1.5))
            else:
                new_bonds.append(new_bond(order=1))
        else:
            new_bonds.append(bond)

    return atoms, new_bonds


def test_valid_chain(chain: ParserChain, prev_aromatic: bool):
    atoms, bonds = apply_stream_parse_method(parse_chain, "a", prev_aromatic)
    assert [atom for atom, _ in atoms[1:]] == chain[0]
    assert bonds == chain[1]


def test_valid_dot_bond_chain():
    # TODO: Implement
    pass
