## Quick orientation for AI coding agents

This repository implements a brain-region classification pipeline built around a small set of Python modules in `src/` and a top-level driver `run.py`.
Keep instructions short and actionable: reference these files and the configuration-driven workflow.

Key entry points
- `run.py` – primary pipeline. Common invocations:
  - `python run.py --sample` (quick smoke run using `data/sample/*`)
  - `python run.py --config config.yaml` (full pipeline)

- `src/data.py` – CSV loading and schema validation. Expects first column `subject_id` and connection columns named `RegionA~RegionB`.
- `src/features.py` – reconstructs symmetric connectivity matrices, imputes diagonals (see `diagonal_strategy`), and builds the per-region classification dataset.
- `src/model.py` – trains a logistic-regression classifier using subject-wise GroupKFold and saves model + scaler to `data/processed`.
- `src/evaluate.py` and `src/visualize.py` – compute error maps, confusion matrices and create publication-quality figures saved under `reports/`.
- `src/utils.py` – config loading, logging, seeding and provenance logging (used by `run.py`).

Project-specific conventions (critical)
- Data schema: CSV with `subject_id` as column 0; all other columns are pairwise connections using `~` as separator. The repository will validate that there are >100 connection columns.
- Single-subject connectivity rows are flattened upper-triangle values; `src/features.py` reconstructs full symmetric matrices and then creates N samples per subject (one per region).
- Diagonal handling: controlled by `config.yaml` `diagonal_strategy` (options: `zero`, `one`, `mean`, `network_mean`). `network_mean` requires region naming conventions used in `parse_networks` inside `src/features.py`.
- Cross-validation: GroupKFold on subject IDs (subject-wise splits) — implemented in `train_classifier`.
- Outputs: processed CSVs + model in `data/processed/`, tables in `reports/tables/`, and figures in `reports/figures/`. Provenance YAML is written to the configured tables directory.

Developer workflows & commands
- Quick smoke test: `python run.py --sample` (fast, creates `data/sample/*` files).
- Full pipeline: `python run.py --config config.yaml` (reads `config.yaml` for paths and hyperparameters).
- Virtualenv: a local venv exists at `brain_connectivity_classifier/` with a Python binary at `brain_connectivity_classifier/bin/python`. Either activate that environment or `pip install -r requirements.txt`.
- Tests: there is a `tests/test_smoke.py` script, but it appears to reference older function names and may be outdated. Prefer `python run.py --sample` as the canonical quick check.

Patterns & pitfalls for contributors
- Prefer using the public helper functions defined in `src/` rather than reimplementing data parsing. Example: use `load_connectivity_data()` then `create_classification_dataset()`.
- Be careful with region name parsing: region/network inference in `parse_networks()` uses specific prefixes like `LH_`,`RH_` and string patterns for subcortical labels — changing names will break `network_mean` diagonal handling and network aggregation.
- I/O paths are configured in `config.yaml`. Avoid hard-coding paths; follow the pattern in `run.py` which loads `config.yaml` and uses `config['output_dirs']`.
- Performance: training uses `n_jobs=-1` in `LogisticRegression` but feature creation reconstructs matrices and expands per-region samples in Python loops — watch memory when running full datasets.

What to look for when editing or extending
- If adding new analysis stages, wire them into `run.py` steps and log provenance using `utils.log_provenance()`.
- If you change column naming conventions, update `src/data.py::validate_schema()` and `src/features.py::extract_regions()` consistently.
- When adding tests, mirror the pipeline usage: create a `tests/` case that runs `python run.py --sample` and asserts expected files are produced (`data/processed/region_list.csv`, `reports/figures/*`). This avoids brittle import-based smoke-tests that the current `tests/test_smoke.py` suffers from.

Quick examples (copyable)
- Run quick sample pipeline:
  python run.py --sample

- Run full pipeline with config:
  python run.py --config config.yaml

Files to reference when coding
- `run.py`, `config.yaml`, `src/data.py`, `src/features.py`, `src/model.py`, `src/evaluate.py`, `src/visualize.py`, `src/utils.py`, `data/sample/`, `data/processed/`, `reports/`

Notes about repository state discovered automatically
- No `.github/copilot-instructions.md` previously existed; this file is newly added.
- The `tests/test_smoke.py` appears to be outdated (referenced function names differ from `src/`) — prefer running `run.py --sample` for verification.

If anything here is unclear or you want deeper detail (example: exact meaning of `network_mean` parsing, memory bounds for full dataset runs, or test updates), tell me which area to expand and I'll update this file.
