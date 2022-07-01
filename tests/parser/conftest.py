#!/usr/bin/env python3

import string

import pytest

from pydepict.parser import (
    BOND_TO_ORDER,
    CHARGE_SYMBOLS,
    ELEMENT_SYMBOLS,
    MIN_CHARGE_MAGNITUDE,
    ORGANIC_SYMBOLS,
    TERMINATORS,
)

NONEXISTENT_ELEMENT_SYMBOLS = []
for first_char in string.ascii_uppercase:
    if first_char in ELEMENT_SYMBOLS:
        continue
    for second_char in string.ascii_lowercase:
        symbol = first_char + second_char
        if symbol not in ELEMENT_SYMBOLS:
            NONEXISTENT_ELEMENT_SYMBOLS.append(symbol)


# Elements
@pytest.fixture(scope="package", params=ELEMENT_SYMBOLS)
def valid_element(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="package", params=NONEXISTENT_ELEMENT_SYMBOLS)
def invalid_element(request: pytest.FixtureRequest) -> str:
    return request.param


# Organic symbols
@pytest.fixture(scope="package", params=ORGANIC_SYMBOLS)
def valid_organic(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(
    scope="package",
    params=[element for element in ELEMENT_SYMBOLS if element not in ORGANIC_SYMBOLS],
)
def valid_element_not_organic(request: pytest.FixtureRequest) -> str:
    return request.param


# Hydrogens
@pytest.fixture(scope="package", params=string.digits)
def valid_hcount(request: pytest.FixtureRequest) -> str:
    return request.param


# Charges
@pytest.fixture(
    scope="package", params=range(-MIN_CHARGE_MAGNITUDE, MIN_CHARGE_MAGNITUDE + 1)
)
def valid_charge(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(scope="package", params=CHARGE_SYMBOLS)
def valid_symbol_charge(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="package", params=CHARGE_SYMBOLS)
def valid_double_symbol_charge(request: pytest.FixtureRequest) -> str:
    return 2 * request.param


# Digits
@pytest.fixture(scope="package", params=string.digits)
def valid_digit(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(
    scope="package",
    params=string.punctuation + string.ascii_lowercase + string.ascii_uppercase,
)
def invalid_digit(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="package", params=range(0, 100, 2))
def valid_double_digits(request: pytest.FixtureRequest) -> str:
    return f"{request.param:0>2}"


# Numbers
@pytest.fixture(
    scope="package",
    params=(
        [num for num in range(10)]
        + [num for num in range(10, 100, 5)]
        + [num for num in range(100, 1000, 50)]
        + [num for num in range(1000, 10000, 500)]
    ),
)
def valid_number(request: pytest.FixtureRequest) -> int:
    return request.param


# Isotopes
@pytest.fixture(scope="package")
def valid_isotope(valid_number: int) -> int:
    return valid_number


# Atom classes
@pytest.fixture(scope="package")
def valid_class(valid_number: int) -> int:
    return valid_number


# Bonds
@pytest.fixture(scope="package", params=BOND_TO_ORDER)
def valid_bond(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(
    scope="package",
    params=[symbol for symbol in string.punctuation if symbol not in BOND_TO_ORDER],
)
def invalid_bond(request: pytest.FixtureRequest) -> str:
    return request.param


# Terminators
@pytest.fixture(scope="package", params=TERMINATORS)
def valid_terminator(request: pytest.FixtureRequest) -> str:
    return request.param
