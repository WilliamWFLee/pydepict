#!/usr/bin/env python3

"""
tests.timing

Measures program execution time for increasing SMILES strings lengths
"""

import timeit
from typing import Tuple

from pydepict import parse, depict, Renderer  # noqa:F401

REPEATING_UNIT = "C(Br)"
MAX_REPEATING_UNITS = 40
NUM_REPEATS = 100


def time_string(smiles: str) -> Tuple[float, float, float]:
    """
    Measures the execution time of parsing, depicting and rendering a SMILES string

    :param smiles: The SMILES string to show
    :type smiles: str
    :return: The average time it takes to parse, depicter,
             and perform render calculations for the given string
    :rtype: Tuple[float, float, float]
    """
    total_parser_time = timeit.timeit(
        stmt=f"graph, _ = parse({smiles!r})",
        number=NUM_REPEATS,
        globals=globals(),
    )

    total_depicter_time = timeit.timeit(
        stmt="positions = depict(graph)",
        setup=f"graph, _ = parse({smiles!r})",
        number=NUM_REPEATS,
        globals=globals(),
    )

    total_renderer_time = timeit.timeit(
        stmt="renderer.set_structure(graph, positions)",
        setup=(
            f"graph, _ = parse({smiles!r})\n"
            "positions = depict(graph)\n"
            "renderer = Renderer()\n"
        ),
        number=NUM_REPEATS,
        globals=globals(),
    )

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
