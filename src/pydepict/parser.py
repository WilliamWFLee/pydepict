#!/usr/bin/env python3

"""
pydepict.parser

Parsing for strings conforming to the OpenSMILES specification
"""

import warnings
from functools import wraps
from typing import Callable, Dict, Generic, Iterable, TypeVar

import networkx as nx

from .consts import CHARGE_SYMBOLS, ELEMENT_FIRST_CHARS, ELEMENTS, AtomAttribute
from .errors import ParserError, ParserWarning

__all__ = ["Stream", "Parser"]

T = TypeVar("T")


class Stream(Generic[T]):
    """
    Stream class for allowing one-item peekahead.

    .. attribute:: pos

        The position within the iterable at which the stream is,
        initially at 0.

        :type: int
    """

    _NO_DEFAULT = object()

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


class Parser:
    """
    Class representing a SMILES parser

    .. attribute:: smiles

        The SMILES string that is parsed by this instance.

        :type: str
    """

    def __init__(self, smiles: str):
        self.smiles = smiles

    def catch_stop_iteration(func: Callable[[], T]) -> Callable[[], T]:
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
        def wrapper(self, *args, **kwargs) -> T:
            try:
                return func(self, *args, **kwargs)
            except StopIteration:
                raise ParserError("Unexpected end-of-stream", self._stream.pos)

        return wrapper

    @catch_stop_iteration
    def parse_number(self) -> int:
        """
        Parse a number (integer) from the stream

        :raise ParserError: If no number is next in stream
        :return: The parsed number
        :rtype: int
        """
        number = ""
        while True:
            try:
                number += self.parse_digit()
            except ParserError:
                break

        if not number:
            raise ParserError(
                f"Expected digit, got {self._stream.peek()}", self._stream.pos
            )

        return int(number)

    @catch_stop_iteration
    def parse_isotope(self) -> int:
        """
        Parses an isotope specification from the stream

        :return: The isotope number parsed
        :rtype: int
        """
        return self.parse_number()

    @catch_stop_iteration
    def parse_element_symbol(self) -> str:
        """
        Parses an element symbol from the stream

        :raises ParserError: If the element symbol is not a known element,
                            or a valid element symbol is not read
        :return: The element parsed
        :rtype: str
        """
        first_char = self._stream.peek()
        if first_char in ELEMENT_FIRST_CHARS:
            element = next(self._stream)
            next_char = self._stream.peek("")
            if next_char and first_char + next_char in ELEMENTS:
                element += next(self._stream)

            if element in ELEMENTS:
                return element

            raise ParserError(f"Invalid element symbol {element!r}", self._stream.pos)

        raise ParserError("Expected element symbol", self._stream.pos)

    @catch_stop_iteration
    def parse_digit(self) -> str:
        """
        Parses a single digit from the stream

        :raises ParserError: If character from stream is not a digit
        :return: The digit parsed
        :rtype: str
        """
        char = self._stream.peek()
        if char.isdigit():
            return next(self._stream)
        raise ParserError(
            f"Expected digit, got {self._stream.peek()}", self._stream.pos
        )

    @catch_stop_iteration
    def parse_hcount(self) -> int:
        """
        Parses hydrogen count from the stream

        :raises ParserError: If the next symbol in the stream is not 'H'
        :return: The hydrogen count, defaults to 0 if at stream end
        :rtype: int
        """
        peek = self._stream.peek(None)
        if peek == "H":
            next(self._stream)
            try:
                count = int(self.parse_digit())
            except ParserError:
                count = 1
            return count
        elif peek is None:
            return 0
        raise ParserError(f"Expected 'H', got {peek!r}", self._stream.pos)

    @catch_stop_iteration
    def parse_charge(self) -> int:
        """
        Parses charge from the given stream

        :return: The charge, defaults to 0 if not found
        :rtype: int
        """
        peek = self._stream.peek(None)
        if peek in CHARGE_SYMBOLS:
            sign = next(self._stream)
            if self._stream.peek(None) == sign:
                next(self._stream)
                warnings.warn(
                    ParserWarning(
                        f"Use of {2 * sign} instead of {sign}2 is deprecated",
                        self._stream.pos,
                    )
                )
                return int(sign + "2")
            try:
                first_digit = self.parse_digit()
            except ParserError:
                return int(sign + "1")
            try:
                second_digit = self.parse_digit()
            except ParserError:
                return int(sign + first_digit)
            return int(sign + first_digit + second_digit)
        if peek is None:
            return 0
        raise ParserError(
            f"Expected charge symbol, got {self._stream.peek()!r}", self._stream.pos
        )

    @catch_stop_iteration
    def parse_atom(self) -> Dict[str, AtomAttribute]:
        """
        Parses the next atom in the stream.

        :return: A dictionary of atom attributes
        :rtype: Dict[str, AtomAttribute]
        """
        attrs = {}
        if self._stream.peek() != "[":
            raise ParserError(
                f"Expected '[' for start of bracket atom, got {self._stream.peek()!r}",
                self._stream.pos,
            )
        next(self._stream)

        try:
            attrs["isotope"] = self.parse_isotope()
        except ParserError:
            attrs["isotope"] = None

        attrs["element"] = self.parse_element_symbol()
        for attr, parse_method, default in [
            ("hcount", self.parse_hcount, 0),
            ("charge", self.parse_charge, 0),
        ]:
            try:
                attrs[attr] = parse_method()
            except ParserError:
                attrs[attr] = default

        if self._stream.peek() != "]":
            raise ParserError(
                f"Expected ']' for end of bracket atom, got {self._stream.peek()!r}",
                self._stream.pos,
            )
        next(self._stream)

        return attrs

    def parse(self) -> nx.Graph:
        """
        Parse the given SMILES string to produce a graph representation.

        :return: The graph represented by the string
        :rtype: nx.Graph
        """

        g = nx.Graph()
        atom_index = 0
        self._stream = Stream(self.smiles)
        while True:
            try:
                peek: str = self._stream.peek()
            except StopIteration:
                break
            if peek == "[":
                g.add_node(atom_index, **self.parse_atom())
                atom_index += 1

        return g
