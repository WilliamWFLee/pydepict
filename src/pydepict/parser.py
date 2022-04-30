#!/usr/bin/env python3

"""
pydepict.parser

Parsing for strings conforming to the OpenSMILES specification

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

import string
import warnings
from functools import wraps
from typing import Iterable, List, Optional, Tuple, Type, TypeVar, Union

import networkx as nx

from .consts import (
    BOND_TO_ORDER,
    CHARGE_SYMBOLS,
    CHIRALITY_CODES,
    CHIRALITY_CODES_FIRST_CHARS,
    CHIRALITY_RANGES,
    DEFAULT_ATOM,
    DEFAULT_BOND,
    ELEMENT_SYMBOL_FIRST_CHARS,
    ELEMENT_SYMBOLS,
    ORGANIC_SYMBOL_FIRST_CHARS,
    ORGANIC_SYMBOLS,
    TERMINATORS,
    VALENCES,
    Atom,
    AtomAttribute,
    AtomRnums,
    Bond,
    BondAttribute,
    Rnum,
    Rnums,
)
from .errors import ParserError, ParserWarning
from .models import Stream
from .utils import atom_valence, is_allenal_center

__all__ = ["parse"]

T = TypeVar("T")
E = TypeVar("E", Type[ParserError], Type[ParserWarning])

# Sentinel object
EXPECT_DEFAULT = object()


def fill_hydrogens(graph: nx.Graph):
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
            possible_valences = list(
                filter(lambda x: x >= current_valence, element_valences)
            )
            if not possible_valences:
                # Hydrogen count is 0 if current valence
                # is already higher than any known valence
                attrs["hcount"] = 0
            target_valence = min(possible_valences)
            attrs["hcount"] = target_valence - current_valence


def imply_shorthand_chirality(graph: nx.Graph):
    """
    Implies the chirality type for atoms in the specified graph
    where there is shorthand chirality specification, i.e. @ or @@

    Shorthand chirality is implied whenever the chirality code is :data:`None`

    :param graph: The graph to scan for shorthand chirality
    :type graph: nx.Graph
    """

    for atom_index, chirality in graph.nodes(data="chirality"):
        if chirality is not None and chirality[0] is None:
            graph.nodes[atom_index]["chirality"] = (
                "AL" if is_allenal_center(atom_index, graph) else "TH"
            )


def infer_bond(atom_aromatic: bool, other_atom_aromatic: bool) -> Bond:
    """
    Returns the inferred bond in the absence of any bond symbol
    based on the relevant atoms.

    Aromatic bond is inferred between two aromatic atoms,
    otherwise single bond is inferred.

    :param atom_aromatic: The aromaticity of one of the atoms
    :type atom_aromatic: bool
    :param other_atom_aromatic: The aromaticity of the other atom
    :type other_atom_aromatic: bool
    :return: A new bond attribute dictionary
    :rtype: Bond
    """
    return new_bond(order=1.5 if atom_aromatic and other_atom_aromatic else 1)


def resolve_rnums(
    atom_rnums: AtomRnums,
    atom_index: int,
    rnums: Rnums,
    graph: nx.Graph,
    stream: Stream,
):
    """
    Resolves a ordered list of rnum specifications for the specified atom,
    using the specified dictionary of unpaired rnums specifications.
    Bonds are formed in the specified graph as required.

    The dictionary of rnums specifications is changed in-place.

    :param atom_rnums: A dictionary of rnums for the specified atom index,
                       mapping of an integer representing the rnum index
                       to a float, representing the associated bond order
    :type atom_rnum: Rnum
    :param atom_index: The index of the atom associated with :param:`atom_rnum`.
    :type atom_index: int
    :param rnums: A dictionary of unpaired rnum specifications,
                  mapping each rnum to a tuple of the index of the atom
                  where the rnum first occurred, and the specified bond order.
    :param graph: The graph to form bonds on
    :type graph: nx.Graph
    :raises ParserError: If a pair of rnums is found, both explicitly state
                         the bond order, and those orders are mismatched;
                         a pair of rnums attempts to join two already bonded atoms;
                         or a pair of rnums attempts to join an atom to itself.
    """
    for rnum_index, bond_order in atom_rnums:
        if rnum_index in rnums:
            # Rnum index already encountered
            other_atom_index, other_bond_order = rnums[rnum_index]

            # Cannot bond atom to itself
            if atom_index == other_atom_index:
                raise new_exception("Cannot bond atom to itself using rnum", stream)
            # Cannot already bond atoms again
            if graph.has_edge(atom_index, other_atom_index):
                raise new_exception("Atoms already bonded", stream)

            # Aim to store bond order in variable 'bond_order'
            if bond_order is not None and other_bond_order is not None:
                if bond_order != other_bond_order:
                    raise ParserError(
                        f"Explicit bond orders for rnum {rnum_index} do not match: "
                        f"{bond_order} and {other_bond_order}"
                    )
            elif bond_order is None and other_bond_order is not None:
                bond_order = other_bond_order
            elif bond_order is None and other_bond_order is None:
                # Imply bond order from atom types
                if (
                    graph.nodes[atom_index]["aromatic"]
                    and graph.nodes[other_atom_index]["aromatic"]
                ):
                    bond_order = 1.5
                else:
                    bond_order = 1

            graph.add_edge(atom_index, other_atom_index, **new_bond(order=bond_order))
            del rnums[rnum_index]
        else:
            # Rnum index not encountered
            rnums[rnum_index] = (atom_index, bond_order)


def new_atom(**attrs: AtomAttribute) -> Atom:
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


def new_bond(**attrs: BondAttribute) -> Bond:
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


def catch_stop_iteration(func):
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
    from the specified iterable :param:`symbols`, otherwise raise :class:`ParserError`.

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
def parse_chiral(stream: Stream[str]) -> Tuple[Optional[str], int]:
    """
    Parses an atomic chirality specification from the specified stream.

    The code is the name given to the two-character alphabetic sequence
    specifying the type of chirality, and the index is the name given
    to the one- or two-digit number that follows the code.

    If shorthand chirality, i.e. @ or @@, is used, the code returned is :data:`None`,
    leaving the code to be determined semantically.

    :param stream: The stream to read from
    :type stream: Stream[str]
    :raises ParserError: If the next symbol in the stream is not '@'
    :return: The chirality specification as a tuple of chirality code and index,
             e.g. ("TH", 1)
    :rtype: Tuple[Optional[str], int]
    """
    expect(stream, "@", "'@'")
    peek = stream.peek("X")

    if peek == "@":
        # @@
        next(stream)
        return None, 2
    if peek not in CHIRALITY_CODES_FIRST_CHARS:
        # @
        return None, 1

    # Code, i.e. TB, AL, SP, TB, OH
    code = next(stream)
    code += stream.peek("")
    if code not in CHIRALITY_CODES:
        raise new_exception(f"Unknown chirality code {code!r}", stream)
    next(stream)

    # Index
    if stream.peek("a") not in string.digits:
        # Must be at least one digit
        raise new_exception("Expected digit", stream)
    index = parse_digit(stream)
    if stream.peek("a") in string.digits:
        # Parse second digit
        index += parse_digit(stream)
    index = int(index)

    if index <= 0 or index > CHIRALITY_RANGES[code]:
        raise new_exception(
            f"Invalid chirality {code}{index}: {index} is out of range", stream
        )

    return code, index


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
    expect(stream, "H", "'H'")
    if stream.peek("a") in string.digits:
        count = int(parse_digit(stream))
    else:
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
    if stream.peek("") == sign:
        next(stream)
        warnings.warn(
            new_exception(
                f"Use of {2 * sign} instead of {sign}2 is deprecated",
                stream,
                ParserWarning,
            )
        )
        return int(sign + "2")
    if stream.peek("a") not in string.digits:
        return int(sign + "1")

    first_digit = parse_digit(stream)
    if stream.peek("a") not in string.digits:
        return int(sign + first_digit)

    second_digit = parse_digit(stream)
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
    if not stream.peek() in string.digits:
        raise new_exception("Expected number for atom class", stream) from None
    return parse_number(stream)


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

    expect(stream, "[")
    if stream.peek() in string.digits:
        attrs["isotope"] = parse_isotope(stream)
    else:
        attrs["isotope"] = None

    attrs["element"] = parse_element_symbol(stream)
    for attr, parse_method, expected_peeks, default in [
        ("chirality", parse_chiral, "@", None),
        ("hcount", parse_hcount, "H", 0),
        ("charge", parse_charge, CHARGE_SYMBOLS, 0),
        ("class", parse_class, ":", None),
    ]:
        if stream.peek() in expected_peeks:
            attrs[attr] = parse_method(stream)
        else:
            attrs[attr] = default

    expect(stream, "]")

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
        attrs = parse_bracket_atom(stream)
        atom.update(**attrs)
    elif stream.peek() in ORGANIC_SYMBOL_FIRST_CHARS:
        # Organic subset symbol
        element = parse_organic_symbol(stream)
        atom["element"] = element
    else:
        raise new_exception("Invalid atom or unknown organic symbol", stream)

    # Deal with aromatic atoms
    if atom["element"].islower():
        atom["element"] = atom["element"].upper()
        atom["aromatic"] = True

    return atom


@catch_stop_iteration
def parse_rnum(stream: Stream[str]) -> int:
    """
    Parses an rnum specification from the specified stream.

    :param stream: The stream to read from
    :type stream: Stream[str]
    :return: The parsed rnum
    :rtype: int
    """
    if stream.peek() in string.digits:
        return int(parse_digit(stream))
    else:
        expect(stream, "%", "percentage sign for rnum")
        return int("".join(parse_digit(stream) for _ in range(2)))


@catch_stop_iteration
def parse_chain(
    stream: Stream[str], prev_is_aromatic: bool = False
) -> Tuple[List[Tuple[Atom, AtomRnums]], List[Bond]]:
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
    :return: A 2-tuple, with the first element list of tuple of atoms
             and a corresponding dictionary of rnums for that atom.
             The second is a list of bonds, in the same order
             as the atoms.
             The number of atoms is always one more than the number of bonds,
             where rnums about the atom preceding the start of the chain
             is stored in the first element of the atoms list
    :rtype: Tuple[List[Atom], List[Bond]]
    """
    atoms = [({"aromatic": prev_is_aromatic}, [])]
    bonds = []
    while True:
        rnum = None
        dot = False
        if stream.peek("") == ".":
            # Dot bond
            next(stream)
            dot = True
            bond = None
        else:
            # Other bonds
            try:
                bond = parse_bond(stream)
            except ParserError:
                bond = None
        # Rnums invalid after dot bond
        if not dot and stream.peek("a") in string.digits + "%":
            # Rnums
            rnum = parse_rnum(stream)
            # The rnum is associated with the most recently parsed atom
            atoms[-1][1].append((rnum, bond))
        if rnum is None:
            # Add parsed bond (for error detection)
            if bond is not None:
                bonds.append(bond)
            # Atoms
            if not stream.peek("") in {"["} | ORGANIC_SYMBOL_FIRST_CHARS:
                break
            atom = parse_atom(stream)
            if not dot and bond is None:
                bond = infer_bond(atom["aromatic"], atoms[-1][0]["aromatic"])
                bonds.append(bond)
            atoms.append((atom, []))
    if len(bonds) + 1 > len(atoms) or not bonds:
        raise new_exception("Expected atom or rnum", stream)
    return atoms, bonds


@catch_stop_iteration
def parse_branch(
    stream: Stream[str], graph: nx.Graph, prev_atom_idx: int, atom_idx: int, rnums: Rnum
):
    line_encountered = False
    expect(stream, "(", "opening parenthesis for branch")
    while True:
        bond = None
        dot = False
        if stream.peek() in BOND_TO_ORDER:
            # Bond
            bond = parse_bond(stream)
        elif stream.peek() == ".":
            # Dot
            next(stream)
            dot = True

        # Line
        if not stream.peek() in {"["} | ELEMENT_SYMBOL_FIRST_CHARS:
            break
        first_atom_idx, atom_idx = parse_line(stream, graph, atom_idx, rnums)
        line_encountered = True

        # Bond previous atom to first atom in line
        if not dot:
            if bond is None:
                bond = infer_bond(
                    graph.nodes[first_atom_idx]["aromatic"],
                    graph.nodes[prev_atom_idx]["aromatic"],
                )
            graph.add_edge(first_atom_idx, prev_atom_idx, **bond)
        prev_atom_idx = atom_idx - 1
    if not line_encountered:
        raise new_exception("Branch cannot be empty", stream)

    expect(stream, ")", "closing parenthesis for branch")
    return atom_idx


@catch_stop_iteration
def parse_line(
    stream: Stream[str], graph: nx.Graph, atom_idx: int, rnums: Rnums
) -> Tuple[int, int]:
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
    :param rnums: A dictionary storing unpaired rnums
    :type rnums: Rnums
    :raises ParserError: If no atom for the start of the line is found
    :return: A tuple of the index for the first atom in the line,
             and the new value for the atom index counter
    :rtype: Tuple[int, int]
    """

    graph.add_node(atom_idx, **parse_atom(stream))
    first_atom_idx = last_chain_atom_idx = atom_idx
    atom_idx += 1
    while True:
        peek = stream.peek(None)
        if peek is None:
            break
        if (
            peek in ("[", "%", ".")
            or peek in BOND_TO_ORDER
            or peek in string.digits
            or peek in ORGANIC_SYMBOL_FIRST_CHARS
        ):
            # Chain
            atoms, bonds = parse_chain(stream, graph.nodes[atom_idx - 1]["aromatic"])
            _, first_atom_rnums = atoms.pop(0)

            resolve_rnums(first_atom_rnums, atom_idx - 1, rnums, graph, stream)

            for (atom, atom_rnums), bond in zip(atoms, bonds):
                graph.add_node(atom_idx, **atom)
                if bond is not None:
                    graph.add_edge(last_chain_atom_idx, atom_idx, **bond)

                resolve_rnums(atom_rnums, atom_idx, rnums, graph, stream)
                last_chain_atom_idx = atom_idx
                atom_idx += 1
        elif peek == "(":
            # Branch
            atom_idx = parse_branch(stream, graph, last_chain_atom_idx, atom_idx, rnums)
        else:
            break

    return first_atom_idx, atom_idx


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
    rnums = {}

    if not smiles:
        raise new_exception("Cannot parse empty string", stream)

    # Syntax parsing + on-the-fly semantics
    parse_line(stream, graph, atom_idx, rnums)
    parse_terminator(stream)

    # Post-parsing semantics

    # Fill implied hydrogens for organic atoms
    fill_hydrogens(graph)
    if rnums:
        # Rnums must be matched
        raise ParserError("Unmatched rnums: " + ", ".join(str(rnum) for rnum in rnums))
    # Determine whether shorthand chirality is tetrahedral or allenal
    imply_shorthand_chirality(graph)

    return graph, get_remainder(stream)
