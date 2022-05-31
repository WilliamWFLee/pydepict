#!/usr/bin/env python3

"""
tests.renderer.utils

Utility functions and values for testing the renderer
"""

import pygame
import pytest

requires_video = pytest.mark.skipif(
    not pygame.display.get_init(), reason="requires video support"
)
