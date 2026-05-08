from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lsh_research.algorithms.bit_sampling_lsh import BitSamplingLSH
from lsh_research.algorithms.lsh_forest import LSHForest
from lsh_research.algorithms.pivot_lsh_forest import PivotLSHForest
from lsh_research.datasets import DatasetBundle, make_mixed_difficulty_bundles


@dataclass
class ExperimentRow:
    seed: int
    dataset: str
    family: str
    n_points: int
    n_queries: int
    dimension: int
    algorithm: str
    pivot_count: int
    num_trees: int
    leaf_size: int
    radius: int
    approximation: float
    build_ms: float
    recall: float
    avg_query_ms: float
    avg_nodes_visited: float
    avg_pivot_comparisons: float
    avg_candidate_comparisons: float
    avg_depth: float
    memory_proxy: int


def evaluate_bundle(bundle, seed, algorithm_name, index, pivot_count, num_trees, leaf_size, approximation):
    successes = 0
    total_ms = 0.0
    total_nodes = 0
    total_pivots = 0
    total_candidates = 0
    total_depth = 0

    threshold = int(np.ceil(bundle.radius * approximation))
    for query in bundle.queries:
        stats = index.query(query, radius = bundle.radius, approximation = approximation)
        total_ms += stats.elapsed_ms
        total_nodes += stats.nodes_visited
        total_pivots += stats.pivot_comparisons
        total_candidates += stats.candidate_comparisons
        total_depth += stats.max_depth_reached
        if stats.success and stats.found_distance is not None and stats.found_distance <= threshold:
            successes += 1

    n_queries = len(bundle.queries)
    return ExperimentRow(seed = seed, dataset = bundle.name, family = bundle.family,
                         n_points = bundle.n_points, n_queries = bundle.n_queries,
                         dimension = bundle.dimension, algorithm = algorithm_name,
                         pivot_count = pivot_count, num_trees = num_trees, leaf_size = leaf_size,
                         radius = bundle.radius, approximation = approximation,
                         build_ms = index.build_elapsed_ms, recall = successes / n_queries,
                         avg_query_ms = total_ms / n_queries,
                         avg_nodes_visited = total_nodes / n_queries,
                         avg_pivot_comparisons = total_pivots / n_queries,
                         avg_candidate_comparisons = total_candidates / n_queries,
                         avg_depth = total_depth / n_queries,
                         memory_proxy = index.memory_proxy()
    )


@dataclass
class SummaryRow:
    dataset: str
    family: str
    n_points: int
    dimension: int
    algorithm: str
    pivot_count: int
    num_trees: int
    leaf_size: int
    radius: int
    approximation: float
    runs: int
    recall_mean: float
    recall_std: float
    avg_query_ms_mean: float
    avg_query_ms_std: float
    build_ms_mean: float
    build_ms_std: float
    avg_nodes_visited_mean: float
    avg_pivot_comparisons_mean: float
    avg_candidate_comparisons_mean: float
    avg_depth_mean: float
    memory_proxy_mean: float


def summarize_rows(rows):
    grouped: dict[tuple[object, ...], list[ExperimentRow]] = {}
    for row in rows:
        key = (
            row.dataset,
            row.family,
            row.n_points,
            row.dimension,
            row.algorithm,
            row.pivot_count,
            row.num_trees,
            row.leaf_size,
            row.radius,
            row.approximation
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[SummaryRow] = []
    for key, group in grouped.items():
        recall_values = [row.recall for row in group]
        query_values = [row.avg_query_ms for row in group]
        build_values = [row.build_ms for row in group]
        summary_rows.append(
            SummaryRow(
                dataset = key[0],
                family = key[1],
                n_points = key[2],
                dimension = key[3],
                algorithm = key[4],
                pivot_count = key[5],
                num_trees = key[6],
                leaf_size = key[7],
                radius = key[8],
                approximation = key[9],
                runs = len(group),
                recall_mean = statistics.fmean(recall_values),
                recall_std = statistics.pstdev(recall_values) if len(group) > 1 else 0.0,
                avg_query_ms_mean = statistics.fmean(query_values),
                avg_query_ms_std = statistics.pstdev(query_values) if len(group) > 1 else 0.0,
                build_ms_mean = statistics.fmean(build_values),
                build_ms_std = statistics.pstdev(build_values) if len(group) > 1 else 0.0,
                avg_nodes_visited_mean = statistics.fmean([row.avg_nodes_visited for row in group]),
                avg_pivot_comparisons_mean = statistics.fmean([row.avg_pivot_comparisons for row in group]),
                avg_candidate_comparisons_mean = statistics.fmean([row.avg_candidate_comparisons for row in group]),
                avg_depth_mean = statistics.fmean([row.avg_depth for row in group]),
                memory_proxy_mean = statistics.fmean([row.memory_proxy for row in group])
            )
        )
    summary_rows.sort(key = lambda row: (row.dataset, row.algorithm, row.pivot_count))
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description = "Run LSH Forest research experiments.")
    parser.add_argument("--output", type = Path, default = Path("results/experiment_results.csv"))
    parser.add_argument("--seeds", type = int, nargs = "+", default = [0, 1, 2])
    parser.add_argument("--approximation", type = float, default = 2.0)
    parser.add_argument("--num-trees", type = int, default = 12)
    parser.add_argument("--pivot-counts", type = int, nargs = "+", default = [1, 2, 4, 8])
    parser.add_argument("--coordinate-count", type = int, default = 16)
    parser.add_argument("--leaf-size", type = int, default = 8)
    args = parser.parse_args()

    rows: list[ExperimentRow] = []

    for seed in args.seeds:
        datasets = make_mixed_difficulty_bundles(seed = seed)
        for bundle in datasets:
            bit_lsh = BitSamplingLSH(points = bundle.points,
                                     coordinate_count = args.coordinate_count,
                                     num_tables = args.num_trees,
                                     seed = seed
            )
            rows.append(
                evaluate_bundle(bundle = bundle,
                                seed = seed,
                                algorithm_name = "bit_sampling_lsh",
                                index = bit_lsh,
                                pivot_count = 0,
                                num_trees = args.num_trees,
                                leaf_size = args.leaf_size,
                                approximation = args.approximation
                )
            )

            forest = LSHForest(points = bundle.points,
                               num_trees = args.num_trees,
                               seed = seed,
                               leaf_size = args.leaf_size
            )
            rows.append(
                evaluate_bundle(bundle = bundle,
                                seed = seed,
                                algorithm_name = "lsh_forest",
                                index = forest,
                                pivot_count = 1,
                                num_trees = args.num_trees,
                                leaf_size = args.leaf_size,
                                approximation = args.approximation
                )
            )

            for pivot_count in args.pivot_counts:
                pivot_forest = PivotLSHForest(points = bundle.points,
                                              num_trees = args.num_trees,
                                              pivot_count = pivot_count,
                                              separation_radius = max(1, int(args.approximation * bundle.radius) - bundle.radius),
                                              seed = seed,
                                              leaf_size = args.leaf_size
                )
                rows.append(
                    evaluate_bundle(bundle = bundle,
                                    seed = seed,
                                    algorithm_name = "pivot_lsh_forest",
                                    index = pivot_forest,
                                    pivot_count = pivot_count,
                                    num_trees = args.num_trees,
                                    leaf_size = args.leaf_size,
                                    approximation = args.approximation
                    )
                )

    args.output.parent.mkdir(parents = True, exist_ok = True)
    with args.output.open("w", newline = "") as fh:
        writer = csv.DictWriter(fh, fieldnames = list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary_rows = summarize_rows(rows)
    summary_path = args.output.with_name(f"{args.output.stem}_summary.csv")
    with summary_path.open("w", newline = "") as fh:
        writer = csv.DictWriter(fh, fieldnames = list(asdict(summary_rows[0]).keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(asdict(row))

    print(f"Wrote {len(rows)} raw rows to {args.output}")
    print(f"Wrote {len(summary_rows)} summary rows to {summary_path}")


if __name__ == "__main__":
    main()
