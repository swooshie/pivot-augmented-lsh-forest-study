# Pivot-Augmented LSH Forest Study

This repository contains the code and final outputs for an empirical study of
the pivot parameter in a simplified pivot-augmented LSH Forest prototype for
approximate nearest neighbor search in Hamming space.

## Repository Structure

- `src/lsh_research/` : implementations and dataset generation
- `experiments/run_experiment.py` : generates raw and summary CSV results
- `experiments/analyze_results.py` : generates figure-ready plots from summary results
- `results/experiment_results.csv` : raw experiment output
- `results/experiment_results_summary.csv` : aggregated summary output
- `results/figures/` : final figures used for the report

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
