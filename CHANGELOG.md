# Changelog

All notable changes to FrontFinder are listed here. The format follows [Keep a Changelog](https://keepachangelog.com),
and versions follow [Semantic Versioning](https://semver.org).

## [2.0.0] - 2026-10-30

Version 2 is a rewrite. The code is now an installable, typed, tested Python package (`fronts`) driven by YAML configs.
1.x models do not load directly in 2.0. Use the `model_1702` harness listed below to run them.

### Added

- **Package layout.** All code lives under `src/fronts/`: `data/` (generation, inputs, targets, datasets), `layers/`
  (losses, metrics, modules), `evaluation/`, `plot/`, `model_1702/`, and the top-level `train`, `evaluate`,
  `calibrate`, `importance`, and `pipeline` modules.
- **YAML configs** in `configs/` for every run: data generation, training, evaluation, ablations, and the pipeline.
- **ERA5 input stores.** `fronts.data.generate` builds an icechunk store from the public Arraylake ERA5 data, with
  derived variables (equivalent potential temperature, virtual temperature, dewpoint, and others), terrain, and a
  choice of pressure levels and time resolution. It can run on SLURM through `dask-jobqueue`.
- **Nine target classes:** no front, cold, warm, stationary, occluded, dryline, trough, tropical trough, and instability
  axis. Forming and dissipating fronts merge into their parent type.
- **2D and 3D UNet3+.** The 3D model treats pressure levels as a depth axis, matching the AIES architecture.
- **Losses:** neighborhood Brier score with latitude-aware windows, multiclass weighted binary cross-entropy, fractions
  skill score, CSI, Brier skill score, and POD, with separate class weights for losses and metrics.
- **Training features:** a `PyDataset` data loader, batch-level shuffling, EMA weight averaging, learning-rate decay,
  early stopping, checkpointing, multi-GPU training with `MirroredStrategy`, and Weights & Biases logging of the full
  config, data shapes and date ranges, per-front-type losses and metrics, and test-set prediction maps.
- **Evaluation:** TP/FP/FN statistics across 50-250 km neighborhoods and 100 probability thresholds, land and ocean
  masks, per-region stats for five forecast-office regions, and performance diagrams.
- **Temperature-scaling calibration** (`fronts.calibrate`) and a calibrated-versus-uncalibrated comparison.
- **Permutation importance** (`fronts.importance`) for ranking input variables.
- **`model_1702` harness** that loads the published AIES FrontFinder model, rebuilds its original inputs, and scores it
  with the same code as 2.0 models.
- **`fronts.pipeline`**, which runs generate, train, and evaluate in order and skips stages whose outputs exist.
- **Tooling:** `pixi` environments with lockfiles (including a `schooner` env for OU's clusters), a `pytest` suite,
  `ruff` and `pyrefly` checks, pre-commit hooks, and GitHub Actions for tests, lint, types, and lockfiles.

### Changed

- Runs on Python 3.12 with TensorFlow 2.16 or later (Keras 3). Models save in the `.keras` format.
- Dependencies are declared in `pyproject.toml` instead of `requirements.txt`.

### Removed

- The 1.x root-level scripts (such as `train_model.py`, `models.py`, and `file_manager.py`) and their custom loss and
  activation files.

## [1.0.0] - 2025-09-18

- FrontFinder workflow 1.0.0.

[2.0.0]: https://github.com/ai2es/fronts/releases/tag/v2.0.0
[1.0.0]: https://github.com/ai2es/fronts/releases/tag/v1.0.0
