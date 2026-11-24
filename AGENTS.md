# FrontFinder (fronts)

## Goal
FrontFinder finds weather fronts in ERA5 reanalysis (a global record of past weather) with a UNet3+ neural network. It is an update of NOAA's FrontFinder. Andrew Justin made the first version. Taylor Mandelbaum made the update. The model labels each grid point with one of nine classes: no front, cold, warm, stationary, occluded, dryline, trough, tropical trough, instability axis.

## Stack
- Language/framework: Python, TensorFlow 2.16+ / Keras 3; typed with `pyrefly`, linted and formatted with `ruff`.
- Environments: `pixi` (HPC, glibc 2.17 on OU Sooner; lock in `pixi.lock`) and `uv` (development and CI; lock in `uv.lock`). Keep both locks in sync. CI checks them.
- Data: ERA5 in Icechunk and zarr stores. `fronts.data.generate` makes them. One YAML file in `configs/` drives each run. The weights (`v2.0.1.keras`) are on Hugging Face (`aaaTman/FrontFinder`). `v2.0.1.keras` takes 27 input channels (no 2 m dewpoint); `v2.0.0.keras` takes 28. They are not interchangeable. Use `configs/sooner_eval_regions_v2.0.1.yaml` for the 27-channel model; every other config builds 28.
- Eval truth: the global eval (`fronts.evaluation.global_eval`, `fronts.data.gml`, `scripts/gml_to_zarr.py`) scores
  against TWC fronts and the global surface analyses behind them. Neither is published, so that path cannot run on
  open data. Its config is not committed either (it names where the rasterized fronts live): the global `scripts/*.sbatch`
  jobs take it as `GLOBAL_CONFIG`. Keep analysis and raster paths out of committed files.
- Compute: training, evaluation, and data generation run on OU Sooner and Schooner (GPU, Slurm). See `scripts/*.sbatch`. Unit tests and lint run locally with no GPU.

## Commands
- Setup: `just setup` (uv dev env) or `pixi install` (HPC env)
- Test: `just test`
- Lint: `just lint` (ruff check, ruff format --check, pyrefly)
- Data: `just generate`
- Train: `just train` / eval: `just eval`
- Deploy: `just deploy` is a stub. Deploy goes through CI (see Do not).

## Conventions
- Python style: format with ruff. Type every function (pyrefly must pass). Write short docstrings. Pre-commit runs ruff and pyrefly.
- Change behavior in the YAML files in `configs/`, not with flags. Keep configs in version control with the code.
- Use one worktree for each task. Open PRs into `master`. CI (`ci.yml`, `tests.yml`, `check-locks.yml`) must pass.
- Never commit data, keys, or model weights.

## Do not
- Edit `uv.lock` or `pixi.lock` by hand.
- Deploy directly. Open a PR and let CI check it.
- Try to run the global eval on open data. The TWC fronts it scores against are not published.

## Current state
`docs/decisions.md` lists decisions and open questions.
