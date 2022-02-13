import pytest

from pydepict.consts import CLOSE_BRACKET, ELEMENTS, OPEN_BRACKET
from pydepict.errors import ParserError
from pydepict.parser import Stream, parse, parse_atom, parse_element


@pytest.mark.parametrize("element", ELEMENTS)
def test_parse_element(element):
    """
    Tests parsing a stream of a single element symbol
    """
    stream = Stream(element)
    result = parse_element(stream)
    assert result == element


@pytest.mark.parametrize("element", ELEMENTS)
def test_parse_element_only_bracket_atom(element):
    """
    Tests parsing a stream of an single element-only SMILES string, e.g ``[Au]``
    """
    # Test parsing atom attributes
    stream = Stream(f"{OPEN_BRACKET}{element}{CLOSE_BRACKET}")
    result = parse_atom(stream)
    assert result["element"] == element

    # Test parsing graph
    stream = Stream(f"{OPEN_BRACKET}{element}{CLOSE_BRACKET}")
    result = parse(stream)
    assert result.nodes[0]["element"] == element


@pytest.mark.parametrize(
    "element", [element.lower() for element in ELEMENTS if element.isalpha()]
)
def test_parse_lowercase_symbols(element):
    """
    Tests parsing lowercase element symbols, which should be rejected.
    """
    with pytest.raises(ParserError):
        stream = Stream(element)
        parse_element(stream)


@pytest.mark.parametrize("element", ("Fg", "Ak", "Of", "My", "Dj"))
def test_parse_nonexistent_symbols(element):
    """
    Tests parsing nonexistent element symbols, which should be rejected.
    """
    with pytest.raises(ParserError):
        stream = Stream(element)
        parse_element(stream)   
