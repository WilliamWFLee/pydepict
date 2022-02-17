#!/usr/bin/env python3

import pytest


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    if item.iter_markers(name="must_pass"):
        if call.excinfo is not None:
            item.parent._mpfailed = item


def pytest_runtest_setup(item: pytest.Item):
    failed_must_pass: pytest.Item = getattr(item.parent, "_mpfailed", None)
    if failed_must_pass is not None:
        pytest.skip(f"{failed_must_pass.name} is marked as must passed.")
