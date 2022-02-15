#!/usr/bin/env python3

"""
pydepict.errors

Custom error classes
"""


class ParserError(Exception):
    """
    Error class for all parser errors
    """
    def __init__(self, msg: str, position: int) -> None:
        """
        Initialises an instance of a parser error

        :param msg: The error message
        :type msg: str
        :param position: The position within the stream at which the error occurred
        :type position: int
        """
        super().__init__(f"{msg}, pos {position}")


class ParserWarning(Warning):
    """
    Warning class for all parser warnings
    """
    def __init__(self, msg: str, position: int) -> None:
        """
        Initialises an instance of a parser warning

        :param msg: The warning message
        :type msg: str
        :param position: The position within the stream at which the error occurred
        :type position: int
        """
        super().__init__(f"{msg}, pos {position}")
