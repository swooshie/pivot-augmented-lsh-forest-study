from __future__ import annotations

import numpy as np

from lsh_research.algorithms.lsh_forest import LSHForest


class PivotLSHForest(LSHForest):
    def __init__(self, points, num_trees, pivot_count, separation_radius, seed = 0, leaf_size = 1):
        self.pivot_count = pivot_count
        self.separation_radius = separation_radius
        super().__init__(points = points, num_trees = num_trees, seed = seed, leaf_size = leaf_size)

    def _choose_pivots(self, indices):
        if indices.size == 0:
            return np.array([], dtype = np.int64)
        subset = self.points[indices]
        center = subset.mean(axis = 0)
        center_distances = np.count_nonzero(np.abs(subset - center) > 0.5, axis = 1)
        order = np.argsort(center_distances, kind = "stable")
        selected: list[int] = []
        threshold = self.separation_radius
        for pos in order:
            candidate_idx = int(indices[pos])
            if not selected:
                selected.append(candidate_idx)
            else:
                candidate = self.points[candidate_idx]
                if all(int(np.count_nonzero(candidate != self.points[prev])) >= threshold for prev in selected):
                    selected.append(candidate_idx)
            if len(selected) >= self.pivot_count:
                break
        return np.array(selected, dtype = np.int64)