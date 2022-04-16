#!/usr/bin/env python3

from setuptools import setup

requirements = []
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(install_requires=requirements)
