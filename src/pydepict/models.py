#!/usr/bin/env python3

"""
pydepict.models

Models for representing data.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

from math import sqrt
from typing import NamedTuple, Tuple


class Vector(NamedTuple):
    x: float
    y: float

    @classmethod
    def from_tuple(cls, coords: Tuple[float, float]) -> "Vector":
        return cls(coords[0], coords[1])

    @staticmethod
    def distance(vector1: "Vector", vector2: "Vector") -> float:
        """
        Calculates the distance between two vectors as if they represented positions
        from a fixed origin.
        """
        return sqrt((vector2.x - vector1.x) ** 2 + (vector2.y - vector1.y) ** 2)

    def normal(self) -> "Vector":
        """
        Calculates the normal to this vector.

        :return: The normal
        :rtype: Vector
        """
        return self.__class__(self.y, -self.x)

    def scale_to(self, magnitude: float) -> "Vector":
        """
        Scales this vector to the specified magnitude, and returns the new vector.

        :param magnitude: The magnitude to scale to
        :type magnitude: float
        :return: The scaled vector
        :rtype: Vector
        """
        if isinstance(magnitude, (float, int)):
            curr_magnitude = sqrt(self.x**2 + self.y**2)
            scale_factor = magnitude / curr_magnitude
            return self.__class__(self.x * scale_factor, self.y * scale_factor)
        return NotImplemented

    def floor(self) -> "Vector":
        """
        Truncates the two components of the vector.

        :return: The new vector with components truncated.
        :rtype: Vector
        """
        return self.__class__(int(self.x), int(self.y))

    def x_reflect(self) -> "Vector":
        """
        Reflects the vector in the x-axis, and returns the reflected vector

        :return: The reflected vector
        :rtype: Vector
        """
        return self.__class__(self.x, -self.y)

    def y_reflect(self) -> "Vector":
        """
        Reflects the vector in the y-axis, and returns the reflected vector

        :return: The reflected vector
        :rtype: Vector
        """
        return self.__class__(-self.x, self.y)

    def __add__(self, other: "Vector") -> "Vector":
        """
        Returns the sum of two vectors
        """
        if isinstance(other, self.__class__):
            return self.__class__(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other: "Vector") -> "Vector":
        """
        Returns the difference of two vectors
        """
        if isinstance(other, self.__class__):
            return self.__class__(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, scalar: float) -> "Vector":
        return self.__class__(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector":
        return self.__mul__(scalar)

    def __neg__(self) -> "Vector":
        return self.__mul__(-1)

    def __eq__(self, other: "Vector") -> bool:
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y
        return NotImplemented

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x}, {self.y})"
