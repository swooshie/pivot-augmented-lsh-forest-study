from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen = True)
class DatasetBundle:
    name: str
    points: np.ndarray
    queries: np.ndarray
    ground_truth: np.ndarray
    radius: int
    family: str
    n_points: int
    n_queries: int
    dimension: int


def hamming_distance(a, b):
    return int(np.count_nonzero(a != b))


def flip_bits(base, flips, rng):
    candidate = base.copy()
    if flips <= 0:
        return candidate
    flip_indices = rng.choice(base.shape[0], size = min(flips, base.shape[0]), replace = False)
    candidate[flip_indices] = 1 - candidate[flip_indices]
    return candidate


def make_random_binary_bundle(n_points = 2000, n_queries = 200, dimension = 128, query_flips = 8, seed = 0):
    rng = np.random.default_rng(seed)
    points = rng.integers(0, 2, size = (n_points, dimension), dtype = np.uint8)
    gt_indices = rng.choice(n_points, size = n_queries, replace = True)
    queries = np.vstack([flip_bits(points[idx], query_flips, rng) for idx in gt_indices]).astype(np.uint8)
    return DatasetBundle(name = "random_binary", points = points, queries = queries, ground_truth = gt_indices.astype(np.int64),
                         radius = query_flips, family = "random_binary", n_points = n_points, n_queries = n_queries,
                         dimension = dimension
    )


def make_clustered_binary_bundle(n_points = 2000, n_queries = 200, dimension = 128, n_clusters = 20, intra_flips = 12,
                                 query_flips = 8, seed = 0):
    rng = np.random.default_rng(seed)
    centers = rng.integers(0, 2, size = (n_clusters, dimension), dtype = np.uint8)
    cluster_ids = rng.choice(n_clusters, size = n_points, replace = True)
    points = np.vstack([flip_bits(centers[cid], intra_flips, rng) for cid in cluster_ids]).astype(np.uint8)
    gt_indices = rng.choice(n_points, size = n_queries, replace = True)
    queries = np.vstack([flip_bits(points[idx], query_flips, rng) for idx in gt_indices]).astype(np.uint8)
    return DatasetBundle(name = "clustered_binary", points = points, queries = queries, 
                         ground_truth = gt_indices.astype(np.int64), radius = query_flips, family = "clustered_binary", 
                         n_points = n_points, n_queries = n_queries, dimension = dimension
    )


def make_sparse_binary_bundle(n_points = 4000, n_queries = 300, dimension = 256, active_bits = 12, query_flips = 4, seed = 0):
    rng = np.random.default_rng(seed)
    points = np.zeros((n_points, dimension), dtype = np.uint8)
    for row in range(n_points):
        active = rng.choice(dimension, size = min(active_bits, dimension), replace = False)
        points[row, active] = 1
    gt_indices = rng.choice(n_points, size = n_queries, replace = True)
    queries = np.vstack([flip_bits(points[idx], query_flips, rng) for idx in gt_indices]).astype(np.uint8)
    return DatasetBundle(name = "sparse_binary", points = points, queries = queries, ground_truth = gt_indices.astype(np.int64),
                         radius = query_flips, family = "sparse_binary", n_points = n_points, n_queries = n_queries,
                         dimension = dimension
    )


def make_mixed_difficulty_bundles(seed = 0):
    return [
        make_random_binary_bundle(n_points = 4000, n_queries = 300, dimension = 256, query_flips = 12, seed = seed),
        make_clustered_binary_bundle(n_points = 4000, n_queries = 300, dimension = 256, n_clusters = 24, intra_flips = 20,
                                     query_flips = 12, seed = seed + 101),
        make_sparse_binary_bundle(n_points = 4000, n_queries = 300, dimension = 256, active_bits = 12, 
                                  query_flips = 4, seed = seed + 202)
    ]
