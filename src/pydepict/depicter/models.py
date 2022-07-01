#!/usr/bin/env python3

"""
pydepict.depicter.models

Models for the depicter.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""


from typing import DefaultDict, Dict, Tuple

from ..models import Vector


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
