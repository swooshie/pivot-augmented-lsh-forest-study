from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class QueryStats:
    found_index: int | None
    found_distance: int | None
    success: bool
    nodes_visited: int = 0
    pivot_comparisons: int = 0
    candidate_comparisons: int = 0
    max_depth_reached: int = 0
    elapsed_ms: float = 0.0


class ANNIndex(Protocol):
    def query(self, query, radius, approximation):
        ...

    def memory_proxy(self):
        ...

    @property
    def build_elapsed_ms(self):
        ...
