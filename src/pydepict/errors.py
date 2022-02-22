#!/usr/bin/env python3

"""
pydepict.errors

Custom error classes
"""


class ParserStateException(Exception):
    """
    Error class used when a parser operation is used
    and the parser is not in a state it is permitted to be used in.
    """

    pass


class ParserError(Exception):
    """
    Error class for all errors that occur when parsing input
    """

    def __init__(self, msg: str, position: int) -> None:
        """
        Initialises an instance of a parser exception

        :param msg: The error message
        :type msg: str
        :param position: The position within the stream at which the error occurred
        :type position: int
        """
        super().__init__(f"{msg}, position {position}")
        self.msg = msg
        self.position = position


class ParserWarning(ParserError, Warning):
    """
    Warning class for all warnings that occur when parsing input
    """

    pass
