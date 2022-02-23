#!/usr/bin/env python3

"""
pydepict.parser

Parsing for strings conforming to the OpenSMILES specification
"""

import string
import warnings
from functools import wraps
from typing import Callable, Generic, Iterable, Optional, Type, TypeVar, Union

import networkx as nx

from .consts import (
    BOND_TO_ORDER,
    CHARGE_SYMBOLS,
    ELEMENT_SYMBOL_FIRST_CHARS,
    ELEMENT_SYMBOLS,
    ORGANIC_SYMBOL_FIRST_CHARS,
    ORGANIC_SYMBOLS,
    TERMINATORS,
    Atom,
)
from .errors import ParserError, ParserStateException, ParserWarning

__all__ = ["Stream", "Parser"]

T = TypeVar("T")
E = TypeVar("E", Type[ParserError], Type[ParserWarning])


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


class Parser:
    """
    Class representing a SMILES parser

    .. attribute:: smiles

        The SMILES string that is parsed by this instance.

        :type: str
    """

    # Sentinel object
    DEFAULT = object()

    def __init__(self, smiles: str):
        self.smiles = smiles

    def _catch_stop_iteration(func: Callable[[], T]) -> Callable[[], T]:
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
                raise self._new_exception("Unexpected end-of-stream")

        return wrapper

    def _new_exception(self, msg: str, exc_type: E = ParserError) -> E:
        stream = getattr(self, "_stream", None)
        if stream is None:
            raise ParserStateException(
                "New parser error cannot be instantiated if stream does not exist"
            )
        return exc_type(msg, stream.pos)

    def expect(
        self,
        symbols: Iterable[str],
        terminal: Optional[str] = None,
        default: Union[str, object] = DEFAULT,
    ) -> Union[str, object]:
        """
        Expect the next string to be any character
        from the specified list :param:`symbols`, otherwise raise :class:`ParserError`.

        If end-of-stream is reached, then return :param:`default` if specified,
        or raise :class:`ParserError`.

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
            peek = self._stream.peek()
        except StopIteration:
            if default != self.DEFAULT:
                return default
            else:
                raise self._new_exception("Unexpected end-of-stream")

        if peek in symbols:
            return next(self._stream)

        expected = (
            terminal
            if terminal is not None
            else ", ".join(repr(symbol) for symbol in symbols)
        )
        msg = f"Expected {expected}, got {self._stream.peek()!r}"
        raise self._new_exception(msg)

    @_catch_stop_iteration
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
            raise self._new_exception(f"Expected number, got {self._stream.peek()}")

        return int(number)

    @_catch_stop_iteration
    def parse_bond(self) -> float:
        """
        Parses a bond symbol from the stream

        :raises ParserError: If invalid bond symbol is encountered
        :return: The bond order of the bond symbol
        :rtype: float
        """
        symbol = self.expect(BOND_TO_ORDER, "bond")
        return BOND_TO_ORDER[symbol]

    @_catch_stop_iteration
    def parse_isotope(self) -> int:
        """
        Parses an isotope specification from the stream

        :return: The isotope number parsed
        :rtype: Optional[int]
        """
        return self.parse_number()

    @_catch_stop_iteration
    def parse_element_symbol(self) -> str:
        """
        Parses an element symbol from the stream

        :raises ParserError: If the element symbol is not a known element,
                             or a valid element symbol is not read
        :return: The element parsed
        :rtype: str
        """
        element = self.expect(ELEMENT_SYMBOL_FIRST_CHARS, "alphabetic character")
        next_char = self._stream.peek("")
        if next_char and element + next_char in ELEMENT_SYMBOLS:
            element += next(self._stream)

        if element in ELEMENT_SYMBOLS:
            return element

        raise self._new_exception(f"Invalid element symbol {element!r}")

    @_catch_stop_iteration
    def parse_digit(self) -> str:
        """
        Parses a single digit from the stream

        :raises ParserError: If character from stream is not a digit
        :return: The digit parsed
        :rtype: str
        """
        return self.expect(string.digits, "digit")

    @_catch_stop_iteration
    def parse_hcount(self) -> int:
        """
        Parses hydrogen count from the stream

        :raises ParserError: If the next symbol in the stream is not 'H'
        :return: The hydrogen count
        :rtype: int
        """
        self.expect(("H",), "'H'")
        try:
            count = int(self.parse_digit())
        except ParserError:
            count = 1

        return count

    @_catch_stop_iteration
    def parse_charge(self) -> int:
        """
        Parses a charge from the stream

        :return: The charge parsed
        :rtype: int
        """
        sign = self.expect(CHARGE_SYMBOLS, "charge sign")
        if self._stream.peek(None) == sign:
            next(self._stream)
            warnings.warn(
                self._new_exception(
                    f"Use of {2 * sign} instead of {sign}2 is deprecated",
                    ParserWarning,
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

    @_catch_stop_iteration
    def parse_class(self) -> int:
        """
        Parses an atom class specification from the stream.

        :raises ParserError: If no atom class specification is found
        :return: The atom class as an :class:`int`
        :rtype: int
        """
        self.expect(":", "colon for atom class")
        try:
            return self.parse_number()
        except ParserError:
            raise self._new_exception("Expected number for atom class") from None

    @_catch_stop_iteration
    def parse_bracket_atom(self) -> Atom:
        """
        Parses a bracket atom from the stream

        :raises ParserError: If the opening and closing bracket is not found,
                             or no element is found
        :return: A dictionary of atom attributes
        :rtype: Atom
        """

        attrs = {}

        self.expect("[", "opening bracket for atom")
        try:
            attrs["isotope"] = self.parse_isotope()
        except ParserError:
            attrs["isotope"] = None

        attrs["element"] = self.parse_element_symbol()
        for attr, parse_method, default in [
            ("hcount", self.parse_hcount, 0),
            ("charge", self.parse_charge, 0),
            ("class", self.parse_class, None),
        ]:
            try:
                attrs[attr] = parse_method()
            except ParserError:
                attrs[attr] = default

        self.expect("]", "closing bracket for atom")

        return attrs

    @_catch_stop_iteration
    def parse_organic_symbol(self) -> str:
        """
        Parses an organic subset symbol from the stream.

        :raises ParserError: If the element symbol is not a known element,
                             not a valid element symbol, or is a valid element symbol
                             that cannot be used in an organic context
        :return: The element parsed
        :rtype: str
        """
        try:
            element = self.expect(ORGANIC_SYMBOL_FIRST_CHARS, "alphabetic character")
        except ParserError:
            if self._stream.peek() in ELEMENT_SYMBOLS:
                raise self._new_exception(
                    f"Element symbol {self._stream.peek()!r} "
                    "cannot be used in an organic context"
                )
            raise
        next_char = self._stream.peek("")
        if next_char:
            if element + next_char in ORGANIC_SYMBOLS:
                element += next(self._stream)
            elif element + next_char in ELEMENT_SYMBOLS:
                raise self._new_exception(
                    f"Element symbol {element + next_char!r} "
                    "cannot be used in an organic context"
                )

        if element in ORGANIC_SYMBOLS:
            return element
        if element in ELEMENT_SYMBOLS:
            raise self._new_exception(
                f"Element symbol {element!r} cannot be used in an organic context"
            )

        raise self._new_exception(f"Invalid element symbol {element!r}")

    @_catch_stop_iteration
    def parse_atom(self) -> Atom:
        """
        Parses an atom in the stream.

        :raises ParserError: If no atom is found
        :return: A dictionary of atom attributes
        :rtype: Atom
        """

        # Default atom attributes
        atom = {
            "isotope": None,
            "hcount": None,
            "charge": 0,
            "class": None,
            "aromatic": False,
        }

        if self._stream.peek() == "[":
            # Bracket atom
            try:
                attrs = self.parse_bracket_atom()
            except ParserError:
                raise self._new_exception("Expected atom") from None
            else:
                atom.update(**attrs)
        else:
            # Organic subset symbol
            try:
                element = self.parse_organic_symbol()
            except ParserError:
                raise self._new_exception("Expected atom") from None
            else:
                atom["element"] = element

        # Deal with aromatic atoms
        if atom["element"].islower():
            atom["element"] = atom["element"].upper()
            atom["aromatic"] = True

        return atom

    @_catch_stop_iteration
    def parse_line(self):
        """
        Parses a line. A line is an atom, atoms that follow it and any branches
        that begin at the same level of nesting as the first atom.

        :raises ParserError: If no atom for the start of the line is found
        """

        self._g.add_node(self._atom_index, **self.parse_atom())
        self._atom_index += 1

    def parse_terminator(self) -> None:
        """
        Parses a terminator.

        :raises ParserError: If terminator is not found, and stream is not at end.
        """
        self.expect(TERMINATORS, "terminator", None)
        return None

    def parse(self) -> nx.Graph:
        """
        Parse the given SMILES string to produce a graph representation.

        :return: The graph represented by the string
        :rtype: nx.Graph
        """
        self._g = nx.Graph()
        self._atom_index = 0
        self._stream = Stream(self.smiles)

        self.parse_line()
        self.parse_terminator()

        try:
            return self._g
        finally:
            del self._g, self._atom_index, self._stream
