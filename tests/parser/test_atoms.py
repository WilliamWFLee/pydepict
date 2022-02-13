#!/usr/bin/env python3

"""
tests.parser.test_atoms

Tests the parsing of atoms together
"""

import pytest

from pydepict.consts import CLOSE_BRACKET, ELEMENTS, OPEN_BRACKET
from pydepict.parser import Stream, parse, parse_atom

BRACKET_ATOM_TEMPLATE = f"{OPEN_BRACKET}{{}}{CLOSE_BRACKET}"


@pytest.mark.parametrize("element", ELEMENTS)
def test_parse_element_only_bracket_atom_attributes(element):
    """
    Tests parsing a stream of a single element-only SMILES string, e.g ``[Au]``
    to test for correct attributes
    """
    # Test parsing atom attributes
    stream = Stream(BRACKET_ATOM_TEMPLATE.format(element))
    result = parse_atom(stream)
    assert result["element"] == element


@pytest.mark.parametrize("element", ELEMENTS)
def test_parse_element_only_bracket_atom_graph(element):
    """
    Tests parsing a stream of an single element-only SMILES string, e.g ``[Au]``
    for correct output graph
    """
    stream = Stream(BRACKET_ATOM_TEMPLATE.format(element))
    result = parse(stream)
    assert result.nodes[0]["element"] == element
