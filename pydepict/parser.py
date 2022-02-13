#!/usr/bin/env python3

"""
pydepict.parser

Parsing for strings conforming to the OpenSMILES specification
"""

from typing import Dict, Iterable, Optional, TypeVar

import networkx as nx

from pydepict.consts import (
    CLOSE_BRACKET,
    ELEMENTS,
    OPEN_BRACKET,
    WILDCARD,
    AtomAttribute,
)

from .errors import ParserError

T = TypeVar("T")


class Stream:
    """
    Stream class for allowing one-item peekahead.

    .. attribute:: pos

        The position within the iterable at which the stream is,
        or -1 if the stream has not been read yet.

        :type: int
    """

    _NO_DEFAULT = object()

    def __init__(self, content: Iterable[T]) -> None:
        self._iter = iter(content)
        self._peek = None
        self.pos = -1

    def __iter__(self) -> "Stream":
        return self

    def __next__(self) -> T:
        next_ = self._peek if self._peek is not None else next(self._iter)
        self._peek = None
        self.pos += 1
        return next_

    def peek(self, default: T = _NO_DEFAULT) -> T:
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
                if default != self._NO_DEFAULT:
                    return default
                raise
        return self._peek


def parse_element_symbol(stream: Stream) -> Optional[str]:
    """
    Parses an element symbol from the stream

    :param stream: The stream to read an element symbol from
    :type stream: Stream
    :raises ParserError: If the element symbol is not a known element,
                         or a valid element symbol is not read
    :return: The element parsed
    :rtype: Optional[str]
    """
    first_char = stream.peek("")
    if (first_char.isalpha() and first_char.isupper()) or stream.peek() == WILDCARD:
        element = next(stream)
        next_char = stream.peek("")
        if next_char.isalpha() and next_char.islower():
            element += next(stream)

        if element in ELEMENTS:
            return element

        raise ParserError(f"Invalid element symbol {element!r}", stream.pos)

    raise ParserError("Expected element symbol", stream.pos)


def parse_atom(stream: Stream) -> Dict[str, AtomAttribute]:
    """
    Parses the next atom in the given stream.

    :param stream: The character string to read from
    :type stream: Stream
    :raises ParserError: If opening and closing brackets are expected, but not encountered # noqa: E501
    :return: A dictionary of atom attributes
    :rtype: Dict[str, AtomAttribute]
    """
    attrs = {}
    if next(stream) != OPEN_BRACKET:
        raise ParserError(
            f"Expected {OPEN_BRACKET} for start of bracket atom, got {next(stream)}",
            stream.pos,
        )

    attrs["element"] = parse_element_symbol(stream)

    if next(stream) != CLOSE_BRACKET:
        raise ParserError(
            f"Expected {CLOSE_BRACKET} for end of bracket atom, got {stream.peek()}",
            stream.pos,
        )

    return attrs


def parse(smiles: str) -> nx.Graph:
    g = nx.Graph()
    atom_index = 0
    stream = Stream(smiles)
    while True:
        try:
            peek: str = stream.peek()
        except StopIteration:
            break
        if peek == OPEN_BRACKET:
            g.add_node(atom_index, **parse_atom(stream))
            atom_index += 1

    return g
