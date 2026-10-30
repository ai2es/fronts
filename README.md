# NOAA FrontFinder v2

FrontFinder finds weather fronts in ERA5 reanalysis data with a UNet3+ neural network. This repository is the updated
version of NOAA's FrontFinder code. Andrew Justin (@andrewjustin) created it, and Taylor Mandelbaum (@aaTman) updated it.

The model labels each grid point with one of nine classes: no front, cold front, warm front, stationary front, occluded
front, dryline, trough, tropical trough, and instability axis. Forming and dissipating fronts are merged into their
parent type.

## What changed from v1

- **YAML configs.** Every run (data, training, evaluation) is driven by one YAML file in `configs/`. The YAML is the
  single source of truth, so there are no shell scripts full of flags to manage, and each config is versioned with the
  code.
- **Typed, linted code.** All code is type-checked with `pyrefly` and linted and formatted with `ruff`. Both run as
  pre-commit hooks.
- **Keras 3 and TensorFlow 2.16+.** The model saves as a `.keras` file, which is
  [easier to manage](https://www.tensorflow.org/tutorials/keras/save_and_load#new_high-level_keras_format) than the old
  formats.
- **icechunk data stores.** Inputs and targets live in [icechunk](https://icechunk.io) stores, which are versioned zarr
  stores.
- **2D or 3D models.** The UNet3+ can treat pressure levels as channels (2D) or as a depth axis (3D convolutions).
- **Experiment tracking.** Training logs losses, per-front-type metrics, and test-set prediction maps to Weights & Biases.
- **`pixi` environments.** FrontFinder trains on OU's supercomputers (schooner and sooner), which pin `glibc` to 2.17.
  `pixi` handles that, mixes conda-forge and PyPI packages in one `pyproject.toml`, and locks every environment. It is
  also much faster than conda. Use `pixi-pack` to move an environment to a machine without `pixi`.

## Installing

Clone the repository and install the default environment:

```
git clone https://github.com/aaTman/fronts.git
cd fronts
pixi install
```

The environments are:

| Environment | Use it for                                                    |
|-------------|---------------------------------------------------------------|
| `default`   | Running tests.                                                |
| `data`      | Building the ERA5 input store.                                |
| `linux`     | Training and evaluation on a generic Linux GPU machine.       |
| `schooner`  | Training and evaluation on OU's clusters (adds version pins). |

All environments target `linux-64`.

## Data

### Inputs (ERA5)

The input store is built from the public ERA5 dataset on Arraylake (`arraylake://earthmover-public/era5`). This
downloads 2007-2025 at 6-hour steps, so it takes a long time:

```
pixi run -e data generate
```

That task uses `configs/generate_icechunk.yaml`. To use your own config, or to run the work on a SLURM cluster
through `dask-jobqueue`:

```
pixi run -e data python -m fronts.data.generate --config /path/to/your.yaml --slurm
```

The config sets the variables, pressure levels, dates, domain, and the output store path (`icechunk_storage_config`).

### Targets (fronts)

The front labels are available on request. Training reads them from the icechunk store named in
`data_config.targets_icechunk_config`.

## Training

```
pixi run -e schooner train
```

That task runs `python -m fronts.train --config configs/sooner_train.yaml`. Call the module directly to use another
config. The config has one section per part of the run: `train_config` (loss, learning rate, epochs), `data_config` (stores,
variables, levels, train/val/test years), `model_config` (UNet3+ shape), `callbacks_config` (early stopping,
checkpoints), and `wandb_config`. 2019 is held out for testing and 2018 for validation. Training uses every visible GPU
through TensorFlow's `MirroredStrategy`.

`configs/sooner_train_3d.yaml` trains the 3D model. Set `data_config.volume_inputs: true` to feed volumes instead of
stacked channels.

To fit a temperature-scaling calibration to a trained model on the validation set:

```
pixi run -e schooner python -m fronts.calibrate --config_path configs/sooner_train.yaml
```

### Full pipeline

`fronts.pipeline` runs generate, train, and evaluate in order. It skips any stage whose outputs already exist, and any
stage whose section is missing from the config:

```
pixi run -e schooner python -m fronts.pipeline --config configs/schooner_pipeline.yaml
```

## Evaluation

Evaluation runs the model over the test period and counts hits, misses, and false alarms within 50-250 km of each
labeled front, across 100 probability thresholds. It writes NetCDF stats files.

| Command                                                                                 | What it does                                                         |
|-----------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| `python -m fronts.evaluate --config_path configs/sooner_eval.yaml [--mask land\|ocean]` | Stats for the full domain, or only land or ocean points.             |
| `python -m fronts.evaluation.region_eval --config_path configs/sooner_eval_regions.yaml` | Stats per region: full domain, land, ocean, and five forecast-office regions. |
| `python -m fronts.evaluation.figures --stats_dir /path/to/stats`                        | Performance diagrams and maps from a stats directory.                |
| `python -m fronts.importance --config_path configs/schooner_importance.yaml`            | Ranks input variables by permutation importance.                     |

Run each one inside an environment, for example `pixi run -e schooner python -m fronts.evaluate ...`. The `eval` task
is a shortcut for the first row: `pixi run -e schooner eval --mask land`.

The `scripts/` folder holds the SLURM batch files used on sooner, such as `scripts/sooner_eval_regions.sbatch`.

### Comparing with the v1 model

`fronts.model_1702` loads the published AIES FrontFinder model (`model_1702`) and rebuilds its original inputs, so it
can be scored with the same code as v2 models. Its configs are in `configs/model_1702/`:

```
python -m fronts.model_1702.store --config configs/model_1702/generate_conus.yaml
python -m fronts.evaluation.region_eval --config_path configs/model_1702/eval_1702_conus.yaml
python -m fronts.model_1702.compare --stats_root /path/to/stats --out comparison.csv
```

## Development

Run the tests:

```
pixi run test
```

CI runs the same tests with `uv` on Python 3.12, 3.13, and 3.14 (`uv sync --group test`, then `uv run pytest`).

Install the pre-commit hooks so `ruff` and `pyrefly` run on every commit:

```
uv run pre-commit install
```
