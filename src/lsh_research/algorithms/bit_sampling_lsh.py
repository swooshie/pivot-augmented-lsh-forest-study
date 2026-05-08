from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from lsh_research.algorithms.base import QueryStats


@dataclass
class BitSamplingLSH:
    points: np.ndarray
    coordinate_count: int
    num_tables: int
    seed: int = 0

    def __post_init__(self):
        build_start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        dim = self.points.shape[1]
        self.coordinate_sets = [
            np.sort(rng.choice(dim, size = self.coordinate_count, replace = False))
            for _ in range(self.num_tables)
        ]
        self.tables: list[dict[tuple[int, ...], list[int]]] = []
        for coords in self.coordinate_sets:
            table: dict[tuple[int, ...], list[int]] = {}
            for idx, point in enumerate(self.points):
                key = tuple(int(x) for x in point[coords])
                table.setdefault(key, []).append(idx)
            self.tables.append(table)
        self._build_elapsed_ms = (time.perf_counter() - build_start) * 1000.0

    def query(self, query, radius, approximation):
        start = time.perf_counter()
        seen: set[int] = set()
        stats = QueryStats(found_index = None, found_distance = None, success = False)
        threshold = int(np.ceil(radius * approximation))

        for coords, table in zip(self.coordinate_sets, self.tables):
            key = tuple(int(x) for x in query[coords])
            candidates = table.get(key, [])
            for idx in candidates:
                if idx in seen:
                    continue
                seen.add(idx)
                stats.candidate_comparisons += 1
                distance = int(np.count_nonzero(self.points[idx] != query))
                if distance <= threshold:
                    stats.found_index = idx
                    stats.found_distance = distance
                    stats.success = True
                    stats.elapsed_ms = (time.perf_counter() - start) * 1000.0
                    return stats

        stats.elapsed_ms = (time.perf_counter() - start) * 1000.0
        return stats

    def memory_proxy(self):
        return sum(len(bucket) for table in self.tables for bucket in table.values())

    @property
    def build_elapsed_ms(self):
        return self._build_elapsed_ms
