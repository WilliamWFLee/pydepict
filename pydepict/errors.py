#!/usr/bin/env python3

"""
pydepict.errors

Custom error classes
"""


class ParserError(Exception):
    def __init__(self, msg: str, position: int) -> None:
        super().__init__(f"{msg}, pos {position}")
