#!/usr/bin/env python3

"""
tests.parser.utils

Utility functions for testing the parser
"""

from typing import Any, Callable, Iterable, Optional, TypeVar, Union
from unittest.mock import DEFAULT, MagicMock

import pytest_mock

from pydepict import parser
from pydepict.consts import Atom
from pydepict.errors import ParserError
from pydepict.models import Stream

T = TypeVar("T")

BRACKET_ATOM_TEMPLATE = "[{isotope}{element}H{hcount}{charge:+}:{class}]"


def apply_stream_parse_method(
    meth: Callable[..., T], value: Union[str, Stream], *args, **kwargs
) -> T:
    """
    Applies an parse method to a new instance of :class:`Stream` for :param:`value`,
    and returns the result.

    Additional positional and keyword arguments are passed to the parse method.

    :param meth: The parse method. Must accept an instance of :class:`Stream`
                 as its first argument.
    :type meth_name: str
    :param value: The string used to instantiate the stream with
    :type value: str
    :return: The value returned from the method
    :rtype: T
    """
    if type(value) == str:
        stream = Stream(value)
    else:
        stream = value

    return meth(stream, *args, **kwargs)


def patch_parse_method(
    mocker: pytest_mock.MockerFixture,
    meth: Callable,
    return_value: Optional[Any] = DEFAULT,
    side_effect: Optional[Union[Callable[[], T], Any, Iterable]] = None,
) -> MagicMock:
    """
    Patches the given :class:`Parser` method using the specified mocker
    with the specified return value and side effect.

    :param mocker: The instance of :class:`pytest_mock.MockerFixture` to use
    :type mocker: pytest_mock.MockerFixture
    :param meth_name: The method name to patch without the ``parse_`` prefix
    :type meth_name: str
    :param return_value: The return value to patch with, defaults to :data:`DEFAULT`
    :type return_value: Optional[Any], optional
    :param side_effect: The side effect to patch with, defaults to None
    :type side_effect: Optional[Union[Callable[[], T], Any]], optional
    :return: The mock created
    :rtype: MagicMock
    """
    try:
        iter(side_effect)
    except TypeError:
        pass
    else:
        side_effect = list(side_effect) + [
            ParserError("Exceeded expected number of calls")
        ]
    mock = mocker.patch.object(parser, meth.__name__)
    mock.return_value = return_value
    mock.side_effect = side_effect

    return mock


def format_bracket_atom(atom: Atom) -> str:
    return BRACKET_ATOM_TEMPLATE.format(**atom)
