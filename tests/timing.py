#!/usr/bin/env python3

"""
tests.timing

Measures program execution time for increasing SMILES strings lengths
"""

import time
from typing import Tuple

from pydepict import Renderer, depict, parse

REPEATING_UNIT = "C(Br)"
MAX_REPEATING_UNITS = 25
NUM_REPEATS = 100


def time_string(smiles: str) -> Tuple[float, float, float]:
    """
    Measures the execution time of parsing, depicting and rendering a SMILES string

    :param smiles: The SMILES string to show
    :type smiles: str
    :return: The average time taken to process a string
    :rtype: float
    """
    total_parser_time = 0
    total_depicter_time = 0
    total_renderer_time = 0
    renderer = Renderer()
    for _ in range(NUM_REPEATS):
        start_time = time.perf_counter()
        graph, _ = parse(smiles)
        time_after_parser = time.perf_counter()
        positions = depict(graph)
        time_after_depicter = time.perf_counter()
        renderer.set_structure(graph, positions)
        end_time = time.perf_counter()

        total_parser_time += time_after_parser - start_time
        total_depicter_time += time_after_depicter - time_after_parser
        total_renderer_time += end_time - time_after_depicter

    return tuple(
        time / NUM_REPEATS
        for time in (total_parser_time, total_depicter_time, total_renderer_time)
    )


def main():
    with open("timings.csv", "w") as f:
        f.write("Length,Parser Time,Depicter Time,Renderer Time\n")
        for n in range(1, MAX_REPEATING_UNITS + 1):
            print(n)
            smiles = n * REPEATING_UNIT
            parser_time, depicter_time, renderer_time = time_string(smiles)
            f.write(f"{n},{parser_time},{depicter_time},{renderer_time}\n")


if __name__ == "__main__":
    main()
