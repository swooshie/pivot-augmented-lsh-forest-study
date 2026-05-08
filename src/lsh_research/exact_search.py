from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExactSearchResult:
    index: int
    distance: int


def all_hamming_distances(points, query):
    return np.count_nonzero(points != query, axis = 1)


def exact_nearest_neighbor(points, query):
    distances = all_hamming_distances(points, query)
    index = int(np.argmin(distances))
    return ExactSearchResult(index = index, distance = int(distances[index]))
