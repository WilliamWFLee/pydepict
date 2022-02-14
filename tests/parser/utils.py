#!/usr/bin/env python3

"""
tests.parser.utils

Utility functions for testing the parser
"""

from typing import Callable, TypeVar

from pydepict.parser import Stream

T = TypeVar("T")


def apply_parse_function(func: Callable[[Stream], T], value: str) -> T:
    """
    Applies a parse function to a string and returns the value

    :param func: The parse function
    :type func: Callable[[Stream], T]
    :param value: The string to apply the function to
    :type value: str
    :return: The value returned from the function
    :rtype: T
    """
    stream = Stream(value)
    return func(stream)
