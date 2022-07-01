#!/usr/bin/env python3

"""
pydepict.models

Models for representing data.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

from math import cos, sin, sqrt
from typing import (
    Callable,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Tuple,
    TypeVar,
    Union,
)

from .consts import THIRTY_DEGS_IN_RADS, VECTOR_NAMES

__all__ = ["Stream", "Matrix", "Vector"]

T = TypeVar("T")


class Stream(Generic[T]):
    """
    Stream class for allowing one-item peekahead.

    :class:`Stream` is an iterable so it can be used in a ``for`` loop::

        for item in stream:
            ...

    or using :func:`next` with a stream.

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

        :param default: Value to return if stream is at end instead,
                        or raises :exc:`StopIteration` if not provided.
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


class _MatrixBase:
    @staticmethod
    def _index_valid(index: float) -> bool:
        return 0 <= index < 2


class _MatrixRowView(_MatrixBase):
    """
    View representing a matrix row.

    Setting and getting values from the row is supported,
    and setting values changes the matrix the row comes from.
    """

    def __init__(self, list: List[float]):
        self._list: List[float] = list

    def _check_index(self, column_index: int):
        if not self._index_valid(column_index):
            raise ValueError("Row index must be between 0 and 1")

    def __getitem__(self, column_index: int) -> float:
        self._check_index(column_index)
        return self._list[column_index]

    def __setitem__(self, column_index: int, value: float):
        self._check_index(column_index)
        self._list[column_index] = value


class Matrix(_MatrixBase):
    """
    Represents a 2x2 matrix.
    """

    def __init__(self, values: List[List[float]] = None):
        self._list: List[List[float]]
        if values is None:
            self._list = [[0 for _ in range(2)] for _ in range(2)]
        else:
            if len(values) != 2 or any(len(row) != 2 for row in values):
                raise TypeError(
                    "Provided values is not of correct shape. " "Must be 2x2 list"
                )
            self._list = values

    @classmethod
    def rotate(cls, angle: float) -> "Matrix":
        """
        Returns a new :class:`Matrix` representing an anticlockwise rotation
        by :param:`angle` radians.

        The values of the matrix are rounded to 10 decimal places,
        because of floating-point limitations.

        :param angle: The angle of the rotation that the new matrix represents
        :type angle: float
        :return: The matrix representing the rotation
        :rtype: Matrix
        """
        cos_theta = round(cos(angle), 10)
        sin_theta = round(sin(angle), 10)
        return cls(
            [
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta],
            ]
        )

    def __getitem__(self, row_index: int) -> _MatrixRowView:
        if not self._index_valid(row_index):
            raise ValueError("Row index must be between 0 and 1")
        return _MatrixRowView(self._list[row_index])

    def __mul__(self, vector: "Vector") -> "Vector":
        return Vector(*(sum(u * v for u, v in zip(row, vector)) for row in self._list))

    def __str__(self) -> str:
        return str(self._list)

    def __repr__(self) -> str:
        return f"Matrix({self._list})"


class Vector(NamedTuple):
    """
    Representation of a 2D vector.

    Supports a variety of operations::

    >>> vec1 = Vector(2, 4)
    >>> vec2 = Vector(3, 9)
    >>> vec1 + vec2   # Vector addition
    Vector(5, 13)
    >>> vec1 - vec2  # Vector difference
    Vector(-1, -5)
    >>> vec1 * vec2  # Component-wise vector multiplication
    Vector(6, 36)
    >>> 2.5 * vec1  # Scalar multiplication
    Vector(5.0, 10.0)
    >>> -vec1  # Vector negation
    Vector(-2, -4)
    >>> vec1 == vec2  # Vector equality
    False
    >>> Vector.distance(vec1, vec2)  # Vector distance
    5.0990195135927845
    >>> abs(vec1)  # Vector magnitude
    4.47213595499958
    >>> vec1.normal()  # Calculates a normal to the vector
    Vector(4, -2)
    >>> vec1.rotate(math.pi)  # Rotates π radians anticlockwise
    Vector(-2, -4)
    >>> vec1.x_reflect()  # Reflection in the x-axis
    Vector(2, -4)
    >>> vec2.y_reflect()  # Reflection in the y-axis
    Vector(-2, 4)

    Two class methods :meth:`min_all` and `:meth:`max_all`
    can also be used to find the vector representing the component-wise
    minimum and maximum vectors respectively from an iterable of vectors.
    """

    x: float
    y: float

    @classmethod
    def from_tuple(cls, coords: Tuple[float, float]) -> "Vector":
        return cls(*coords)

    @classmethod
    def _minmax_all(
        cls, vectors: Iterable["Vector"], func: Callable[[Iterable[float]], float]
    ) -> "Vector":
        if func not in (min, max):
            raise ValueError("func must be 'max' or 'min'")
        all_vectors = [vector for vector in vectors]

        if all_vectors:
            folded_x = func(vector.x for vector in all_vectors)
            folded_y = func(vector.y for vector in all_vectors)
        else:
            folded_x = folded_y = 0

        return cls(folded_x, folded_y)

    @classmethod
    def max_all(cls, vectors: Iterable["Vector"]) -> "Vector":
        """
        Calculates the vector that has the maximum value of each component
        in the iterable of vectors provided.

        :param vectors: An iterable of vectors
        :type vectors: Iterable[Vector]
        :return: The vector where each component has the maximum for that component
        :rtype: Vector
        """

        return cls._minmax_all(vectors, max)

    @classmethod
    def min_all(cls, vectors: Iterable["Vector"]) -> "Vector":
        """
        Calculates the vector that has the minimum value of each component
        in the iterable of vectors provided.

        :param vectors: An iterable of vectors
        :type vectors: Iterable[Vector]
        :return: The vector where each component has the minimum for that component
        :rtype: Vector
        """

        return cls._minmax_all(vectors, min)

    @staticmethod
    def distance(vector1: "Vector", vector2: "Vector") -> float:
        """
        Calculates the distance between two vectors as if they represented positions
        from a fixed origin.
        """
        return sqrt((vector2.x - vector1.x) ** 2 + (vector2.y - vector1.y) ** 2)

    def normal(self) -> "Vector":
        """
        Calculates a normal to this vector.

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
            curr_magnitude = abs(self)
            scale_factor = magnitude / curr_magnitude
            return self.__class__(self.x * scale_factor, self.y * scale_factor)
        return NotImplemented

    def rotate(self, angle: float) -> "Vector":
        """
        Rotates this vector by :param:`angle` radians anticlockwise.

        :param angle: The angle to rotate by
        :type angle: float
        :return: The rotated vector
        :rtype: Vector
        """
        return Matrix.rotate(angle) * self

    def floor(self) -> "Vector":
        """
        Truncates the two components of the vector.

        :return: The new vector with components truncated.
        :rtype: Vector
        """
        return self.__class__(int(self.x), int(self.y))

    def x_reflect(self) -> "Vector":
        """
        Reflects the vector in the x-axis, and returns the reflected vector.

        :return: The reflected vector
        :rtype: Vector
        """
        return self.__class__(self.x, -self.y)

    def y_reflect(self) -> "Vector":
        """
        Reflects the vector in the y-axis, and returns the reflected vector.

        :return: The reflected vector
        :rtype: Vector
        """
        return self.__class__(-self.x, self.y)

    def copy(self) -> "Vector":
        """
        Returns a copy of this vector.

        :return: A copy of this vector
        :rtype: Vector
        """
        return self.__class__(self.x, self.y)

    def __abs__(self) -> float:
        return sqrt(self.x**2 + self.y**2)

    def __add__(self, other: "Vector") -> "Vector":
        """
        Returns the sum of two vectors.
        """
        if isinstance(other, self.__class__):
            return self.__class__(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other: "Vector") -> "Vector":
        """
        Returns the difference of two vectors.
        """
        if isinstance(other, self.__class__):
            return self.__class__(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, other: Union[float, "Vector"]) -> "Vector":
        if isinstance(other, (float, int)):
            return self.__class__(self.x * other, self.y * other)
        elif isinstance(other, self.__class__):
            # Component-wise multiplication
            return self.__class__(self.x * other.x, self.y * other.y)

    def __rmul__(self, other: Union[float, "Vector"]) -> "Vector":
        return self.__mul__(other)

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


# Adds constant vectors to :class:`Vector`
for i, name in enumerate(VECTOR_NAMES):
    setattr(Vector, name, Vector(1, 0).rotate(THIRTY_DEGS_IN_RADS * i))
