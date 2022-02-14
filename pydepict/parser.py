#!/usr/bin/env python3

"""
pydepict.parser

Parsing for strings conforming to the OpenSMILES specification
"""

import warnings
from functools import wraps
from typing import Callable, Dict, Generic, Iterable, Optional, TypeVar

import networkx as nx

from .consts import CHARGE_SYMBOLS, ELEMENTS, AtomAttribute
from .errors import ParserError, ParserWarning

T = TypeVar("T")


class Stream(Generic[T]):
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


def catch_stop_iteration(func: Callable[[Stream], T]) -> Callable[[Stream], T]:
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
    def wrapper(stream) -> T:
        try:
            return func(stream)
        except StopIteration:
            raise ParserError("Unexpected end-of-stream", stream.pos)

    return wrapper


@catch_stop_iteration
def parse_element_symbol(stream: Stream[str]) -> Optional[str]:
    """
    Parses an element symbol from the stream

    :param stream: The stream to read an element symbol from
    :type stream: Stream
    :raises ParserError: If the element symbol is not a known element,
                         or a valid element symbol is not read
    :return: The element parsed
    :rtype: Optional[str]
    """
    first_char = stream.peek()
    if (first_char.isalpha() and first_char.isupper()) or stream.peek() == "*":
        element = next(stream)
        next_char = stream.peek("")
        if next_char.isalpha() and next_char.islower():
            element += next(stream)

        if element in ELEMENTS:
            return element

        raise ParserError(f"Invalid element symbol {element!r}", stream.pos)

    raise ParserError("Expected element symbol", stream.pos)


@catch_stop_iteration
def parse_digit(stream: Stream[str]) -> str:
    """
    Parses a single digit from the given stream

    :param stream: The stream to read from
    :type stream: Stream
    :raises ParserError: If character from stream is not a digit
    :return: The digit parsed
    :rtype: str
    """
    char = stream.peek()
    if char.isdigit():
        return next(stream)
    raise ParserError(f"Expected digit, got {stream.peek()}", stream.pos)


@catch_stop_iteration
def parse_hcount(stream: Stream[str]) -> int:
    """
    Parses hydrogen count from the given stream

    :param stream: The stream to read from
    :type stream: Stream
    :return: The hydrogen count, defaults to 0 if not found
    :rtype: int
    """
    if stream.peek(None) == "H":
        next(stream)
        try:
            count = int(parse_digit(stream))
        except ParserError:
            count = 1
        return count
    return 0


@catch_stop_iteration
def parse_charge(stream: Stream[str]) -> int:
    """
    Parses charge from the given stream
    :param stream: The stream to read from
    :type stream: Stream
    :return: The charge, defaults to 0 if not found
    :rtype: int
    """
    if stream.peek(None) in CHARGE_SYMBOLS:
        sign = next(stream)
        if stream.peek(None) == sign:
            next(stream)
            warnings.warn(
                ParserWarning(
                    f"Use of {2 * sign} instead of {sign}2 is deprecated", stream.pos
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
    return 0


@catch_stop_iteration
def parse_atom(stream: Stream[str]) -> Dict[str, AtomAttribute]:
    """
    Parses the next atom in the given stream.

    :param stream: The character string to read from
    :type stream: Stream
    :raises ParserError: If opening and closing brackets are expected, but not encountered # noqa: E501
    :return: A dictionary of atom attributes
    :rtype: Dict[str, AtomAttribute]
    """
    attrs = {}
    if stream.peek() != "[":
        raise ParserError(
            f"Expected '[' for start of bracket atom, got {stream.peek()!r}",
            stream.pos,
        )
    next(stream)

    attrs["element"] = parse_element_symbol(stream)
    attrs["hcount"] = parse_hcount(stream)
    attrs["charge"] = parse_charge(stream)

    if stream.peek() != "]":
        raise ParserError(
            f"Expected ']' for end of bracket atom, got {stream.peek()!r}",
            stream.pos,
        )
    next(stream)

    return attrs


def parse(smiles: str) -> nx.Graph:
    """
    Parse the given SMILES string to produce a graph representation.

    :param smiles: The SMILES string to parse
    :type smiles: str
    :return: The graph represented by the string
    :rtype: nx.Graph
    """

    g = nx.Graph()
    atom_index = 0
    stream = Stream(smiles)
    while True:
        try:
            peek: str = stream.peek()
        except StopIteration:
            break
        if peek == "[":
            g.add_node(atom_index, **parse_atom(stream))
            atom_index += 1

    return g
