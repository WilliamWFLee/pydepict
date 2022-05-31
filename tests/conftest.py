#!/usr/bin/env python3

import pytest


def get_root_ancestor(item: pytest.Item) -> pytest.Item:
    parent = item.parent
    while True:
        if parent.parent is None:
            break
        parent = parent.parent

    return parent


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    # Set must-pass test item if a must-pass test has failed
    if item.get_closest_marker(name="must_pass"):
        if call.excinfo is not None:
            get_root_ancestor(item)._mpfailed = item


def pytest_runtest_setup(item: pytest.Item):
    # Skip test if must-pass test failed
    ancestor = get_root_ancestor(item)
    if getattr(ancestor, "_mpfailed", None) is not None:
        pytest.skip(f"{ancestor._mpfailed.name} is marked as must passed, but failed")
