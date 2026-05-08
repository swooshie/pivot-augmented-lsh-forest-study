from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

MPL_DIR = Path(".mplconfig")
MPL_DIR.mkdir(exist_ok = True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_DIR.resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")



def _method_order_key(row):
    if row["algorithm"] == "bit_sampling_lsh":
        return (0, 0)
    if row["algorithm"] == "lsh_forest":
        return (1, 0)
    return (2, int(row["pivot_count"]))


def _method_label(row):
    if row["algorithm"] == "bit_sampling_lsh":
        return "Bit LSH"
    if row["algorithm"] == "lsh_forest":
        return "Forest"
    return f"Pivot k = {int(row['pivot_count'])}"


def _pivot_label(row):
    if row["algorithm"] == "lsh_forest":
        return "0"
    return str(int(row["pivot_count"]))


def make_method_tradeoff_figure(df, output_dir):
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 2, figsize = (13, 10))
    if len(datasets) == 1:
        axes = [axes]

    for row_index, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset].copy()
        subset = subset.sort_values(by=["algorithm", "pivot_count"], key = None)
        subset = subset.iloc[sorted(range(len(subset)), key = lambda i: _method_order_key(subset.iloc[i]))]
        labels = [_method_label(row) for _, row in subset.iterrows()]
        x = range(len(labels))

        query_ax = axes[row_index][0]
        build_ax = axes[row_index][1]

        query_ax.bar(x, subset["avg_query_ms_mean"], color = "#4C78A8")
        query_ax.set_title(f"{dataset.replace('_', ' ').title()} : Mean Query Time")
        query_ax.set_ylabel("Milliseconds")
        query_ax.set_xticks(list(x), labels, rotation = 30, ha = "right")
        for x_pos, (_, row) in zip(x, subset.iterrows()):
            query_ax.text(x_pos,
                          row["avg_query_ms_mean"],
                          f"rec = {row['recall_mean']:.3f}",
                          ha = "center",
                          va = "bottom",
                          fontsize = 8
            )

        build_ax.bar(x, subset["build_ms_mean"], color = "#F58518")
        build_ax.set_title(f"{dataset.replace('_', ' ').title()} : Mean Build Time")
        build_ax.set_ylabel("Milliseconds")
        build_ax.set_xticks(list(x), labels, rotation = 30, ha = "right")

    fig.tight_layout()
    fig.savefig(output_dir / "figure_method_tradeoffs.png", dpi = 220, bbox_inches = "tight")
    plt.close(fig)


def make_pivot_scaling_figure(df, output_dir):
    forest_rows = df[df["algorithm"].isin(["lsh_forest", "pivot_lsh_forest"])].copy()
    datasets = sorted(forest_rows["dataset"].unique())

    fig, axes = plt.subplots(len(datasets), 3, figsize = (15, 10))
    if len(datasets) == 1:
        axes = [axes]

    colors = {"query": "#4C78A8", 
              "build": "#F58518",
              "memory": "#54A24B"
    }

    for row_index, dataset in enumerate(datasets):
        subset = forest_rows[forest_rows["dataset"] == dataset].copy()
        subset["k_value"] = subset.apply(lambda row: 0 if row["algorithm"] == "lsh_forest" else int(row["pivot_count"]), axis = 1)
        subset = subset.sort_values("k_value")

        axes[row_index][0].plot(subset["k_value"], subset["avg_query_ms_mean"], marker = "o", color = colors["query"])
        axes[row_index][0].set_title(f"{dataset.replace('_', ' ').title()} : Query Time vs k")
        axes[row_index][0].set_xlabel("Pivot count k")
        axes[row_index][0].set_ylabel("Milliseconds")

        axes[row_index][1].plot(subset["k_value"], subset["build_ms_mean"], marker = "o", color = colors["build"])
        axes[row_index][1].set_title(f"{dataset.replace('_', ' ').title()} : Build Time vs k")
        axes[row_index][1].set_xlabel("Pivot count k")
        axes[row_index][1].set_ylabel("Milliseconds")

        axes[row_index][2].plot(subset["k_value"], subset["memory_proxy_mean"], marker = "o", color = colors["memory"])
        axes[row_index][2].set_title(f"{dataset.replace('_', ' ').title()} : Memory Proxy vs k")
        axes[row_index][2].set_xlabel("Pivot count k")
        axes[row_index][2].set_ylabel("Proxy Size")
        axes[row_index][2].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1e6:.2f}M"))

    fig.tight_layout()
    fig.savefig(output_dir / "figure_pivot_scaling.png", dpi = 220, bbox_inches = "tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description = "Generate paper-ready figures from summary results.")
    parser.add_argument("summary_csv", type = Path)
    parser.add_argument("--output-dir", type = Path, default = Path("results/figures"))
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    args.output_dir.mkdir(parents = True, exist_ok = True)
    make_method_tradeoff_figure(df, args.output_dir)
    make_pivot_scaling_figure(df, args.output_dir)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()