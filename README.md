# Pivot-Augmented LSH Forest Study

Authors:

- Aditya Jhaveri (`aaj6301`)
- Bhavik Patwa (`bnp7995`)

This repository contains the code and report for an empirical study of the pivot
parameter in a simplified pivot-augmented LSH Forest prototype for approximate
nearest neighbor search in Hamming space.

The project implements and compares three methods:

- bit-sampling LSH
- vanilla LSH Forest
- pivot-augmented LSH Forest

## Repository Structure

- `src/lsh_research/` : implementations and dataset generation
- `experiments/run_experiment.py` : generates raw and summary CSV results
- `experiments/analyze_results.py` : generates figure-ready plots from summary results
- `latex/main.tex` : LaTeX source for the written report
- `Algorithms_ML_DS_project.pdf` : final compiled report
- `requirements.txt` : Python dependencies

Generated experiment outputs are written under `results/`. This directory is
ignored by git because the results can be reproduced from the scripts.

## Reproducing Results

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=src
python experiments/run_experiment.py --output results/experiment_results.csv
python experiments/analyze_results.py results/experiment_results_summary.csv
```

## Outputs

The experiment script writes :

- a per-seed raw results file
- an aggregated summary file with means and standard deviations

The analysis script writes :

- `results/figures/figure_method_tradeoffs.png`
- `results/figures/figure_pivot_scaling.png`

## Report

The final report is included as `Algorithms_ML_DS_project.pdf`. The LaTeX source
is in `latex/main.tex`.
