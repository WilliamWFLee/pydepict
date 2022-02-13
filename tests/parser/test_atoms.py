#!/usr/bin/env python3

"""
tests.parser.test_atoms

Tests the parsing of atoms and various related aspects,
e.g. elements, charge, isotopes, etc.
"""

import pytest

from pydepict.consts import CLOSE_BRACKET, ELEMENTS, OPEN_BRACKET
from pydepict.errors import ParserError
from pydepict.parser import Stream, parse, parse_atom, parse_element

BRACKET_ATOM_TEMPLATE = f"{OPEN_BRACKET}{{}}{CLOSE_BRACKET}"


@pytest.mark.parametrize("element", ELEMENTS)
def test_parse_element(element):
    """
    Tests parsing a stream of a single element symbol
    """
    stream = Stream(element)
    result = parse_element(stream)
    assert result == element


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


@pytest.mark.parametrize(
    "element", [element.lower() for element in ELEMENTS if element.isalpha()]
)
def test_parse_lowercase_symbols(element):
    """
    Tests parsing lowercase element symbols, which should return :data:`None`
    """
    stream = Stream(element)
    result = parse_element(stream)
    assert result is None


@pytest.mark.parametrize("element", ("Fg", "Ak", "Of", "My", "Dj"))
def test_parse_nonexistent_symbols(element):
    """
    Tests parsing nonexistent element symbols, which should raise an error
    """
    stream = Stream(element)
    with pytest.raises(ParserError):
        result = parse_element(stream)
