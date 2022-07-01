#!/usr/bin/env python3

"""
pydepict.models

Models for representing data.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""

from math import cos, sin, sqrt
from typing import (
    Callable,
    DefaultDict,
    Dict,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")

__all__ = ["Stream", "Matrix", "Vector"]


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


class DepictionConstraints:
    """
    Implements an endpoint order-independent data structure
    for storing chosen constraints, with weights for each atom.

    Setting and retrieving constraints from the data structure
    is via dictionary-style access, with a 2-tuple representing
    the endpoints used as the key::

    >>> constraints[1, 2] = Vector(1, 4)
    >>> constraints[1, 2]
    Vector(1, 4)

    You can use ``in`` to determine if a pair of endpoints
    has a vector constraint associated with it::

    >>> (1, 2) in constraints
    True

    You can also an individual pair of constraints from the data structure::

    >>> del constraints[1, 2]
    >>> (1, 2) in constraints
    False

    or clear the data structure of all vector constraints::
    >>> constraints.clear()

    A dictionary :attr:`weights` is used for storing the sampling weight
    for the constraints chosen for each atom::

    >>> constraints.weights[1] = 0.7

    The underlying data structure for constraints is a dictionary of dictionaries,
    where the keys of both the outer dictionary and inner dictionaries
    are endpoint indices, and the values of the inner dictionaries
    are the vector constraints.

    The 2-tuple key is always sorted before any operation
    on the underlying data structure, such no key in the outer dictionary
    is greater than the keys of its corresponding inner dictionary.

    .. attribute:: weights

        A dictionary of selection weights used during the sampling process
        for choosing atom constraints.

        :type: DefaultDict[int, float]
    """

    def __init__(self):
        self._dict: DefaultDict[int, Dict[int, Vector]] = DefaultDict(lambda: {})
        self.weights: DefaultDict[int, float] = DefaultDict(lambda: 1)

    @staticmethod
    def _sort_key(key: Tuple[int, int]) -> Tuple[Tuple[int, int], bool]:
        """
        Sorts a pair of keys, and returns the sorted keys
        with a :class:`bool` indicating whether the key order has been swapped.

        :param key: The pair of keys as a tuple
        :type key: Tuple[int, int]
        :return: A 2-tuple, the first element a tuple of the keys in sorted order,
                 the second element :data:`True` or :data:`False`,
                 indicating whether the keys have been swapped or not.
        :rtype: Tuple[Tuple[int, int], bool]
        """
        u, v = key
        if u > v:
            return (v, u), True
        return key, False

    def __contains__(self, key: Tuple[int, int]) -> bool:
        (u, v), _ = self._sort_key(key)
        return u in self._dict and v in self._dict[u]

    def __getitem__(self, key: Tuple[int, int]) -> "Vector":
        (u, v), flipped = self._sort_key(key)
        if self.__contains__((u, v)):
            return -self._dict[u][v] if flipped else self._dict[u][v]
        raise KeyError(key)

    def __setitem__(self, key: Tuple[int, int], value: "Vector"):
        (u, v), flipped = self._sort_key(key)
        self._dict[u][v] = -value if flipped else value

    def __delitem__(self, key: Tuple[int, int]):
        if self.__contains__(key):
            (u, v), _ = self._sort_key(key)
            del self._dict[u][v]
        else:
            raise KeyError(key)

    def clear(self):
        """
        Clears all constraints
        """
        self._dict.clear()


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
