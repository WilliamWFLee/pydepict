#!/usr/bin/env python3

"""
pydepict.parser

Parsing for strings conforming to the OpenSMILES specification
"""

import string
import warnings
from functools import wraps
from typing import (
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import networkx as nx

from pydepict.utils import atom_valence

from .consts import (
    BOND_TO_ORDER,
    CHARGE_SYMBOLS,
    DEFAULT_ATOM,
    DEFAULT_BOND,
    ELEMENT_SYMBOL_FIRST_CHARS,
    ELEMENT_SYMBOLS,
    ORGANIC_SYMBOL_FIRST_CHARS,
    ORGANIC_SYMBOLS,
    TERMINATORS,
    VALENCES,
    Atom,
    Bond,
)
from .errors import ParserError, ParserWarning

__all__ = ["Stream", "parse"]

T = TypeVar("T")
E = TypeVar("E", Type[ParserError], Type[ParserWarning])

# Sentinel object
EXPECT_DEFAULT = object()


class Stream(Generic[T]):
    """
    Stream class for allowing one-item peekahead.

    .. attribute:: pos

        The position within the iterable at which the stream is,
        initially at 0.

        :type: int
    """

    DEFAULT = object()

    def __init__(self, content: Iterable[T]) -> None:
        self._iter = iter(content)
        self._peek = None
        self.pos = 0

    def __iter__(self) -> "Stream":
        return self

    def __next__(self) -> T:
        next_ = self._peek if self._peek is not None else next(self._iter)
        self._peek = None
        self.pos += 1
        return next_

    def peek(self, default: T = DEFAULT) -> T:
        """
        Returns the next item in the stream without advancing the stream.

        If stream is at end then return :param:`default`.

        :param default: Value to return if stream is at end instead
        :type: T
        :return: The next item in the stream
        :rtype: T
        """
        if self._peek is None:
            try:
                self._peek = next(self._iter)
            except StopIteration:
                if default != self.DEFAULT:
                    return default
                raise
        return self._peek


def new_atom(**attrs) -> Atom:
    """
    Create new atom attributes dictionary from default atom attributes template.

    Keyword arguments can be used to override defaults.
    Raises :class:`KeyError` if any keyword attributes do not exist
    """
    atom = DEFAULT_ATOM.copy()
    for attr, value in attrs.items():
        if attr not in atom:
            raise KeyError(attr)
        atom[attr] = value
    return atom


def new_bond(**attrs) -> Bond:
    """
    Create new bond attributes dictionary from default bond attributes template.

    Keyword arguments can be used to override defaults.
    Raises :class:`KeyError` if any keyword attributes do not exist
    """
    bond = DEFAULT_BOND.copy()
    for attr, value in attrs.items():
        if attr not in bond:
            raise KeyError(attr)
        bond[attr] = value
    return bond


def catch_stop_iteration(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for methods that throw :class:`StopIteration`.
    Wraps the method such that the exception is caught, and :class:`ParserError`
    is thrown instead.

    :param func: The function to decorate
    :type func: Callable[[Stream], T]
    :return: The decorated function
    :rtype: Callable[[Stream], T]
    """

    @wraps(func)
    def wrapper(stream, *args, **kwargs) -> T:
        try:
            return func(stream, *args, **kwargs)
        except StopIteration:
            raise new_exception("Unexpected end-of-stream", stream)

    return wrapper


def new_exception(msg: str, stream: Stream, exc_type: E = ParserError) -> E:
    """
    Instantiates a new parser exception (default) or warning with the specified message,
    from the specified stream.

    :param msg: The exception message
    :type msg: str
    :param stream: The stream to use
    :type stream: Stream
    :param exc_type: The exception class to use, defaults to ParserError
    :type exc_type: E, optional
    :return: The new exception
    :rtype: E
    """
    return exc_type(msg, stream.pos)


def expect(
    stream: Stream[str],
    symbols: Iterable[str],
    terminal: Optional[str] = None,
    default: Union[str, object] = EXPECT_DEFAULT,
) -> Union[str, object]:
    """
    Expect the next string in the specified stream to be any character
    from the specified list :param:`symbols`, otherwise raise :class:`ParserError`.

    If end-of-stream is reached, then return :param:`default` if specified,
    or raise :class:`ParserError`.

    :param stream: The stream to read from
    :type stream: Stream[str]
    :param symbols: An iterable of symbols to expect
    :type symbols: Iterable[str]
    :param terminal: Name of the terminal to expect, used for error raising
    :type terminal: Optional[str]
    :param default: Value to return if end-of-stream is reached
    :type default: Union[str, object]
    :raises ParserError: If next symbol in stream is not an expected symbol
    :return: The symbol from :param:`symbols` encountered.
    :rtype: Iterable[str]
    """
    try:
        peek = stream.peek()
    except StopIteration:
        if default != EXPECT_DEFAULT:
            return default
        else:
            raise new_exception("Unexpected end-of-stream", stream)

    if peek in symbols:
        return next(stream)

    expected = (
        terminal
        if terminal is not None
        else ", ".join(repr(symbol) for symbol in symbols)
    )
    msg = f"Expected {expected}, got {stream.peek()!r}"
    raise new_exception(msg, stream)


@catch_stop_iteration
def parse_number(stream: Stream[str]) -> int:
    """
    Parse a number (integer) from the specified stream

    :param stream: The stream to read from
    :type stream: Stream[str]
    :raise ParserError: If no number is next in stream
    :return: The parsed number
    :rtype: int
    """
    number = ""
    while True:
        try:
            number += parse_digit(stream)
        except ParserError:
            break

    if not number:
        raise new_exception(f"Expected number, got {stream.peek()}", stream)

    return int(number)


@catch_stop_iteration
def parse_bond(stream: Stream[str]) -> Bond:
    """
    Parses a bond symbol from the specified stream

    :param stream: The stream to read from
    :type stream: Stream[str]
    :raises ParserError: If invalid bond symbol is encountered
    :return: A dictionary of attributes for the parsed bond
    :rtype: Bond
    """
    symbol = expect(stream, BOND_TO_ORDER, "bond")
    return new_bond(order=BOND_TO_ORDER[symbol])


@catch_stop_iteration
def parse_isotope(stream: Stream[str]) -> int:
    """
    Parses an isotope specification from the specified stream

    :param stream: The stream to read from
    :type stream: Stream[str]
    :return: The isotope number parsed
    :rtype: Optional[int]
    """
    return parse_number(stream)


@catch_stop_iteration
def parse_element_symbol(stream) -> str:
    """
    Parses an element symbol from the specified stream

    :param stream: The stream to read from
    :type stream: Stream[str]
    :raises ParserError: If the element symbol is not a known element,
                            or a valid element symbol is not read
    :return: The element parsed
    :rtype: str
    """
    element = expect(stream, ELEMENT_SYMBOL_FIRST_CHARS, "alphabetic character")
    next_char = stream.peek("")
    if next_char and element + next_char in ELEMENT_SYMBOLS:
        element += next(stream)

    if element in ELEMENT_SYMBOLS:
        return element

    raise new_exception(f"Invalid element symbol {element!r}", stream)


@catch_stop_iteration
def parse_digit(stream: Stream[str]) -> str:
    """
    Parses a single digit from the specified stream

    :param stream: The stream to read from
    :type stream: Stream[str]
    :raises ParserError: If character from stream is not a digit
    :return: The digit parsed
    :rtype: str
    """
    return expect(stream, string.digits, "digit")


@catch_stop_iteration
def parse_hcount(stream: Stream[str]) -> int:
    """
    Parses hydrogen count from the specified stream

    :param stream: The stream to read from
    :type stream: Stream[str]
    :raises ParserError: If the next symbol in the stream is not 'H'
    :return: The hydrogen count
    :rtype: int
    """
    expect(stream, ("H",), "'H'")
    try:
        count = int(parse_digit(stream))
    except ParserError:
        count = 1

    return count


@catch_stop_iteration
def parse_charge(stream: Stream[str]) -> int:
    """
    Parses a charge from the specified stream

    :param stream: The stream to read from
    :type stream: Stream[str]
    :return: The charge parsed
    :rtype: int
    """
    sign = expect(stream, CHARGE_SYMBOLS, "charge sign")
    if stream.peek(None) == sign:
        next(stream)
        warnings.warn(
            new_exception(
                f"Use of {2 * sign} instead of {sign}2 is deprecated",
                stream,
                ParserWarning,
            )
        )
        return int(sign + "2")
    try:
        first_digit = parse_digit(stream)
    except ParserError:
        return int(sign + "1")
    try:
        second_digit = parse_digit(stream)
    except ParserError:
        return int(sign + first_digit)
    return int(sign + first_digit + second_digit)


@catch_stop_iteration
def parse_class(stream: Stream[str]) -> int:
    """
    Parses an atom class specification from the specified stream.

    :param stream: The stream to read from
    :type stream: Stream[str]
    :raises ParserError: If no atom class specification is found
    :return: The atom class as an :class:`int`
    :rtype: int
    """
    expect(stream, ":", "colon for atom class")
    try:
        return parse_number(stream)
    except ParserError:
        raise new_exception("Expected number for atom class", stream) from None


@catch_stop_iteration
def parse_bracket_atom(stream: Stream[str]) -> Atom:
    """
    Parses a bracket atom from the specified stream

    :param stream: The stream to read from
    :type stream: Stream[str]
    :raises ParserError: If the opening and closing bracket is not found,
                            or no element is found
    :return: A dictionary of atom attributes
    :rtype: Atom
    """

    attrs = {}

    expect(stream, "[", "opening bracket for atom")
    try:
        attrs["isotope"] = parse_isotope(stream)
    except ParserError:
        attrs["isotope"] = None

    attrs["element"] = parse_element_symbol(stream)
    for attr, parse_method, default in [
        ("hcount", parse_hcount, 0),
        ("charge", parse_charge, 0),
        ("class", parse_class, None),
    ]:
        try:
            attrs[attr] = parse_method(stream)
        except ParserError:
            attrs[attr] = default

    expect(stream, "]", "closing bracket for atom")

    return attrs


@catch_stop_iteration
def parse_organic_symbol(stream: Stream[str]) -> str:
    """
    Parses an organic subset symbol from the specified stream.

    :param stream: The stream to read from
    :type stream: Stream[str]
    :raises ParserError: If the element symbol is not a known element,
                            not a valid element symbol, or is a valid element symbol
                            that cannot be used in an organic context
    :return: The element parsed
    :rtype: str
    """
    try:
        element = expect(stream, ORGANIC_SYMBOL_FIRST_CHARS, "alphabetic character")
    except ParserError:
        if stream.peek() in ELEMENT_SYMBOLS:
            raise new_exception(
                f"Element symbol {stream.peek()!r} "
                "cannot be used in an organic context",
                stream,
            )
        raise
    next_char = stream.peek("")
    if next_char:
        if element + next_char in ORGANIC_SYMBOLS:
            element += next(stream)
        elif element + next_char in ELEMENT_SYMBOLS:
            raise new_exception(
                f"Element symbol {element + next_char!r} "
                "cannot be used in an organic context",
                stream,
            )

    if element in ORGANIC_SYMBOLS:
        return element
    if element in ELEMENT_SYMBOLS:
        raise new_exception(
            f"Element symbol {element!r} cannot be used in an organic context", stream
        )

    raise new_exception(f"Invalid element symbol {element!r}", stream)


@catch_stop_iteration
def parse_atom(stream: Stream[str]) -> Atom:
    """
    Parses an atom in the specified stream.

    :param stream: The stream to read from
    :type stream: Stream[str]
    :raises ParserError: If no atom is found
    :return: A dictionary of atom attributes
    :rtype: Atom
    """

    # Default atom attributes
    atom = new_atom()

    if stream.peek() == "[":
        # Bracket atom
        try:
            attrs = parse_bracket_atom(stream)
        except ParserError:
            raise new_exception("Expected atom", stream) from None
        else:
            atom.update(**attrs)
    else:
        # Organic subset symbol
        try:
            element = parse_organic_symbol(stream)
        except ParserError:
            raise new_exception("Expected atom", stream) from None
        else:
            atom["element"] = element

    # Deal with aromatic atoms
    if atom["element"].islower():
        atom["element"] = atom["element"].upper()
        atom["aromatic"] = True

    return atom


@catch_stop_iteration
def parse_chain(
    stream: Stream[str], prev_is_aromatic: bool = False
) -> Tuple[List[Atom], List[Bond]]:
    """
    Parses a chain from the specified stream.

    A chain is composed of consecutive bonded atoms,
    (which may be the dot bond) without branching.
    Must be associated with a preceding atom.

    :param stream: The stream to read from
    :type stream: Stream[str]
    :param prev_is_aromatic: Whether the atom preceding this chain
                                is aromatic or not. Defaults to :data:`False`.
    :rtype: bool
    :return: A tuple of a list of atoms and list of bonds between them,
                in parse order. The number of atoms and bonds are equal.
    :rtype: Tuple[List[Atom], List[Bond]]
    """
    atoms = [{"aromatic": prev_is_aromatic}]
    bonds = []
    while True:
        # Determine bond order
        try:
            bond = parse_bond(stream)
        except ParserError:
            bond = None
        try:
            atom = parse_atom(stream)
        except ParserError:
            break
        if bond is None:
            if atom["aromatic"] and atoms[-1]["aromatic"]:
                bond = new_bond(order=1.5)
            else:
                bond = new_bond(order=1)
        atoms.append(atom)
        bonds.append(bond)
    if not bonds:
        raise new_exception("Expected atom", stream)
    return atoms[1:], bonds


@catch_stop_iteration
def parse_line(stream: Stream[str], graph: nx.Graph, atom_idx: int) -> int:
    """
    Parses a line from the specified stream, and extends the specified graph
    with the new line.

    A line is an atom, atoms that follow it and any branches
    that begin at the same level of nesting as the first atom.

    :param stream: The stream to read from
    :type stream: Stream[str]
    :param graph: The graph to add the new nodes to
    :type graph: nx.Graph
    :param atom_idx: The atom index to initially number new nodes from
    :type atom_idx: int
    :raises ParserError: If no atom for the start of the line is found
    :return: The next atom index for the next node after this line
    """

    graph.add_node(atom_idx, **parse_atom(stream))
    atom_idx += 1
    while True:
        try:
            chain = parse_chain(stream, graph.nodes[atom_idx - 1]["aromatic"])
        except ParserError:
            break
        for atom, bond in zip(*chain):
            graph.add_node(atom_idx, **atom)
            graph.add_edge(atom_idx - 1, atom_idx, **bond)
            atom_idx += 1
    return atom_idx


def parse_terminator(stream: Stream[str]):
    """
    Parses a terminator.

    :param stream: The stream to read from
    :type stream: Stream[str]
    :raises ParserError: If terminator is not found, and stream is not at end.
    """
    expect(stream, TERMINATORS, "terminator", None)


def get_remainder(stream: Stream[str]) -> str:
    """
    Exhausts the rest of the specified stream, and returns the string from it.

    :param stream: The stream to read from
    :type stream: Stream[str]
    :return: The string with the remaining characters from the stream
    :rtype: str
    """
    return "".join(stream)


def fill_hydrogens(graph: nx.Graph) -> str:
    """
    Fills the hcount attribute for atoms where it is :data:`None`
    (implies the atom is organic subset).

    :param graph: The graph to fill hydrogens for.
    :type graph: nx.Graph
    """
    for atom_index, attrs in graph.nodes(data=True):
        if attrs["hcount"] is None:
            element = attrs["element"]
            # Get all "normal" valences for the current atom
            element_valences = VALENCES[element]
            if element_valences is None:
                continue
            current_valence = atom_valence(atom_index, graph)
            # Possible valences must be at least the current valence of the atom
            possible_valencies = list(
                filter(lambda x: x >= current_valence, element_valences)
            )
            if not possible_valencies:
                # Hydrogen count is 0 if current valence
                # is already higher than any known valence
                graph.nodes[atom_index]["hcount"] = 0
            target_valence = min(possible_valencies)
            graph.nodes[atom_index]["hcount"] = target_valence - current_valence


def parse(smiles: str) -> Tuple[nx.Graph, str]:
    """
    Parse the specified SMILES string to produce a graph representation.

    :param smiles: The SMILES string to parse
    :type smiles: str
    :return: A tuple of the graph represented by the SMILES string,
                and the remainder of the SMILEs after the terminator
    :rtype: Tuple[nx.Graph, str]
    """
    stream = Stream(smiles)
    graph = nx.Graph()
    atom_idx = 0

    # Syntax parsing + on-the-fly semantics
    parse_line(stream, graph, atom_idx)
    parse_terminator(stream)

    # Post-parsing semantics
    fill_hydrogens(graph)

    return graph, get_remainder(stream)
