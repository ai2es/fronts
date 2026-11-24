# Changelog

All notable changes to FrontFinder are listed here. The format follows [Keep a Changelog](https://keepachangelog.com).

Versions track the model: `vX.Y.Z` of this package is the release whose weights are published as `vX.Y.Z.keras` on
[Hugging Face](https://huggingface.co/aaaTman/FrontFinder), so the model and the code that runs it always move together.
Each entry below is the code side of that model release.

## [Unreleased]

### Added

- **Neighborhood calibration.** `fronts.neighborhood_calibration` fits an isotonic regression (a fit that never
  decreases) to the 50-250 km reliability curves from `fronts.evaluate`. It maps a raw probability to the chance of a
  front within that radius. The AIES paper used this scale for its maps. The module is a port of the legacy
  `calibrate_model.py` with two changes. Each bin is weighted by its pixel count. Pixels forecast below 0.01 get
  their own bin. The case-study plot uses the table when `predict_config.calibration_path` is set.
- Eval stats now include `nbhd_truth_{ft}`. It is the weighted count of pixels within each radius of a front.

## [2.0.1] - 2026-11-24

This release goes with the **`v2.0.1.keras` weights** on
[Hugging Face](https://huggingface.co/aaaTman/FrontFinder/blob/main/v2.0.1.keras), which are the model to use with it.
The architecture, output classes, built-in normalization layer, and terrain input are all as in 2.0.0, but the model is
trained **without the 2 m dewpoint input**: its input stack is 27 channels instead of 28 (the 2.0.0 channel order with
the dewpoint channel dropped), so inference needs no dewpoint field. The two checkpoints are not interchangeable —
`v2.0.0.keras` still expects 28 channels. `configs/sooner_eval_regions_v2.0.1.yaml` is the 27-channel eval config for
the new model; every other config in `configs/` still builds the 28-channel stack.

The code side is global evaluation against the TWC global surface analysis, plus the SLURM and Weights & Biases
helpers that came out of running it. No changes to config schema or checkpoint format.

**Note on data availability.** Neither the TWC fronts nor the global surface analyses behind them are published, so
`fronts.data.gml`, `scripts/gml_to_zarr.py`, and `fronts.evaluation.global_eval` cannot run on open data. They are in
the release so the scoring the team does internally lives with the code it scores, and nothing else in this release
needs them.

### Added

- **Global evaluation.** `fronts.evaluation.global_eval` scores a checkpoint against the TWC global surface analysis.
  Because the model was trained on northern-hemisphere fronts, each timestep is run twice: `direct` (inputs as they
  are, scored over lat >= 0) and `flipped` (inputs mirrored about the equator with the meridional wind negated and
  the predictions mirrored back, scored over lat < 0). Their union, the `composite`, is the model's global skill and
  the only result written. Time is processed in chunks so predictions and targets stay in memory.
- **GML analysis reader.** `fronts.data.gml` parses TWC surface-analysis GML files and rasterizes their `FRONT`
  features onto the model's lat/lon grid using the original front codes, so the existing `remap_fronts` path applies
  unchanged. Unrecognized `FRONT_TYPE` strings are counted and skipped, never guessed.
- **`scripts/gml_to_zarr.py`**, which rasterizes a directory of GML files into one zarr of front codes in parallel,
  chunked one timestep at a time.
- **v2.0.1 eval config.** `configs/sooner_eval_regions_v2.0.1.yaml` scores the no-dewpoint run over the USAD office
  regions. It is `configs/sooner_eval_regions.yaml` with `2m_dewpoint_temperature` dropped and the run name swapped,
  and a test pins the two together so they cannot drift apart.
- **Global data config**: `configs/generate_icechunk_global.yaml` (global ERA5 inputs, same variables and levels as
  the regional store). The global eval's own config is deliberately not committed, because it names the checkpoint,
  the ERA5 store, and where the rasterized TWC fronts live: the three global SLURM scripts take it as `GLOBAL_CONFIG`,
  and `GlobalEvalConfig` in `fronts.evaluation.global_eval` documents every field it needs.
- **SLURM scripts**: `scripts/sooner_eval_global.sbatch`, a one-week smoke test
  (`scripts/sooner_eval_global_smoke.sbatch`), and `scripts/sooner_eval_no_dewpoint.sbatch`, which scores the 2.0.1
  (no 2 m dewpoint) model on both USAD regions and the globe. Each takes your own global config as `GLOBAL_CONFIG`
  and the no-dewpoint one derives the 27-channel variant from it. Each stage skips its work when its outputs already
  exist, so a cancelled job resumes where it stopped.
- **Weights & Biases helpers**: `scripts/wandb_finalize_run.py` tags and summarizes a finished run, and
  `scripts/wandb_backfill_from_csv.py` re-logs epochs missing from a run using the training run's
  `metrics_epoch.csv`, which is written independently of W&B. Dry run by default.

### Changed

- `fronts.evaluate.accumulate_stats` takes `spatial`, which skips the per-pixel TP/FP/TN/FN counts. On a global grid
  those are (n_fronts, lat, lon, 5, 100) float32 arrays and dominate both memory and run time; when skipped,
  `spatial_ds` is `None` and `compute_derived_stats` omits `spatial_csi_{ft}`. Defaults to `True`, so existing
  callers are unaffected.
- `scripts/wandb_finalize_run.py` makes `--name` optional when `--run_id` is given.

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

[2.0.1]: https://github.com/ai2es/fronts/releases/tag/v2.0.1
[2.0.0]: https://github.com/ai2es/fronts/releases/tag/v2.0.0
[1.0.0]: https://github.com/ai2es/fronts/releases/tag/v1.0.0
