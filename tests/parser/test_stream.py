#!/usr/bin/env python3

"""
tests.parser.test_charges

Tests the stream class used for parsing
"""

import pytest

from pydepict.parser import Stream


TEST_STRING = "test string"


@pytest.fixture
def stream():
    return Stream(TEST_STRING)


@pytest.mark.run(order=1)
@pytest.mark.must_pass
class TestStream:
    @staticmethod
    def test_stream_sequence(stream: Stream):
        """
        Tests that the iterable represented by the stream is the same
        as the original iterable.
        """
        value = "".join(c for c in stream)
        assert value == TEST_STRING

    @staticmethod
    def test_stream_peek(stream: Stream):
        """
        Tests stream peeking
        """
        # Subsequent peeks should be equal
        first_peek = stream.peek()
        second_peek = stream.peek()
        assert first_peek == second_peek

        # Next call should be same as peek value
        next_ = next(stream)
        assert next_ == first_peek

    @staticmethod
    def test_stream_next(stream: Stream):
        """
        Tests next on stream
        """
        for result, expected in zip(stream, TEST_STRING):
            assert result == expected
