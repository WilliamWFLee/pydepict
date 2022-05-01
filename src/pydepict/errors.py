#!/usr/bin/env python3

"""
pydepict.errors

Custom error classes

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""


from typing import Optional


class ParserError(Exception):
    """
    Error class for all errors that occur when parsing input
    """

    def __init__(self, msg: str, position: Optional[int] = None) -> None:
        """
        Initialises an instance of a parser exception

        :param msg: The error message
        :type msg: str
        :param position: The position within the stream at which the error occurred,
                         or :data:`None` if not applicable. Defaults to :data:`None`
        :type position: Optional[int]
        """
        super().__init__(
            msg + (f", position {position}" if position is not None else "")
        )
        self.msg = msg
        self.position = position


class ParserWarning(ParserError, Warning):
    """
    Warning class for all warnings that occur when parsing input
    """

    pass


class DepicterError(Exception):
    """
    Class for all depicter-related exceptions
    """

    pass
