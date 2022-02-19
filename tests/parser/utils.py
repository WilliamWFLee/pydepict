#!/usr/bin/env python3

"""
tests.parser.utils

Utility functions for testing the parser
"""

from typing import Callable, TypeVar, Union

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
