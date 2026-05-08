from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from lsh_research.algorithms.base import QueryStats


@dataclass
class TreeNode:
    indices: np.ndarray
    split_coordinate: int | None = None
    children: dict[int, "TreeNode"] = field(default_factory = dict)
    pivots: np.ndarray | None = None
    depth: int = 0
    coordinate_position: int = 0


class LSHForest:
    def __init__(self, points, num_trees, seed = 0, leaf_size = 1):
        build_start = time.perf_counter()
        self.points = points
        self.num_trees = num_trees
        self.seed = seed
        self.leaf_size = leaf_size
        self.rng = np.random.default_rng(seed)
        self.dimension = points.shape[1]
        self.coordinate_orders = [
            self.rng.permutation(self.dimension).astype(np.int64) for _ in range(num_trees)
        ]
        all_indices = np.arange(points.shape[0], dtype = np.int64)
        self.trees = [
            self._build_tree(all_indices, order = order, depth = 0, coordinate_position = 0)
            for order in self.coordinate_orders
        ]
        self._build_elapsed_ms = (time.perf_counter() - build_start) * 1000.0

    def _choose_pivots(self, indices):
        if indices.size == 0:
            return np.array([], dtype = np.int64)
        return np.array([int(indices[0])], dtype = np.int64)

    def _build_tree(
        self,
        indices: np.ndarray,
        order: np.ndarray,
        depth: int,
        coordinate_position: int
    ) -> TreeNode:
        node = TreeNode(indices = indices, pivots = self._choose_pivots(indices), 
                        depth = depth, coordinate_position = coordinate_position
        )
        if indices.size <= self.leaf_size or coordinate_position >= self.dimension:
            return node

        split_coordinate = None
        next_position = coordinate_position
        zero_indices = np.array([], dtype = np.int64)
        one_indices = np.array([], dtype = np.int64)
        while next_position < self.dimension:
            candidate_coordinate = int(order[next_position])
            zero_mask = self.points[indices, candidate_coordinate] == 0
            one_mask = ~zero_mask
            candidate_zero = indices[zero_mask]
            candidate_one = indices[one_mask]
            if candidate_zero.size != indices.size and candidate_one.size != indices.size:
                split_coordinate = candidate_coordinate
                zero_indices = candidate_zero
                one_indices = candidate_one
                break
            next_position += 1

        if split_coordinate is None:
            return node
        node.split_coordinate = split_coordinate
        node.coordinate_position = next_position
        node.children[0] = self._build_tree(zero_indices, order, depth + 1, next_position + 1)
        node.children[1] = self._build_tree(one_indices, order, depth + 1, next_position + 1)
        return node

    def _query_tree(self, node, query, threshold, stats):
        current = node
        while True:
            stats.nodes_visited += 1
            stats.max_depth_reached = max(stats.max_depth_reached, current.depth)
            for idx in current.pivots if current.pivots is not None else []:
                stats.pivot_comparisons += 1
                distance = int(np.count_nonzero(self.points[idx] != query))
                if distance <= threshold:
                    stats.found_index = int(idx)
                    stats.found_distance = distance
                    stats.success = True
                    return
            if current.split_coordinate is None:
                for idx in current.indices:
                    stats.candidate_comparisons += 1
                    distance = int(np.count_nonzero(self.points[idx] != query))
                    if distance <= threshold:
                        stats.found_index = int(idx)
                        stats.found_distance = distance
                        stats.success = True
                        return
                return
            branch = int(query[current.split_coordinate])
            child = current.children.get(branch)
            if child is None:
                return
            current = child

    def query(self, query, radius, approximation):
        start = time.perf_counter()
        threshold = int(np.ceil(radius * approximation))
        stats = QueryStats(found_index = None, found_distance = None, success = False)
        for tree in self.trees:
            self._query_tree(tree, query, threshold, stats)
            if stats.success:
                break
        stats.elapsed_ms = (time.perf_counter() - start) * 1000.0
        return stats

    def memory_proxy(self):
        total = 0
        stack = list(self.trees)
        while stack:
            node = stack.pop()
            total += int(node.indices.size)
            total += 0 if node.pivots is None else int(node.pivots.size)
            stack.extend(node.children.values())
        return total

    @property
    def build_elapsed_ms(self):
        return self._build_elapsed_ms