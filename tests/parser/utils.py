#!/usr/bin/env python3

"""
tests.parser.utils

Utility functions for testing the parser
"""

from typing import Any, Callable, Iterable, Optional, TypeVar, Union
from unittest.mock import DEFAULT, MagicMock

import pytest_mock

from pydepict.errors import ParserError
from pydepict.parser import Parser, Stream

T = TypeVar("T")


def apply_parse_method(meth: Callable[[Parser], T], value: Union[str, Stream]) -> T:
    """
    Applies an unbound parse method to a string or :class:`Stream` and returns the value

    :param meth: The parse method
    :type meth: Callable[[Parser], T]
    :param value: The string to apply the function to
    :type value: str
    :return: The value returned from the function
    :rtype: T
    """
    parser = Parser("")
    if type(value) == str:
        stream = Stream(value)
    else:
        stream = value
    parser._stream = stream
    return meth(parser)


def patch_parse_method(
    mocker: pytest_mock.MockerFixture,
    meth_name: str,
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
        side_effect = list(side_effect) + [ParserError("", -1)]
    mock = mocker.patch.object(Parser, f"parse_{meth_name}")
    mock.return_value = return_value
    mock.side_effect = side_effect

    return mock
