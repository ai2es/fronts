"""Orchestrate the full fronts training pipeline: generate → train → eval.

Each stage is skipped automatically when its outputs already exist:
  - Generate: skipped when the icechunk store already contains all requested variables and times.
  - Train: skipped when the model checkpoint file already exists.
  - Eval: skipped when stats output files already exist in the output directory.

Stages can also be disabled entirely by omitting their config section from the YAML.

Usage:
    pixi run python src/fronts/pipeline.py --config configs/schooner_pipeline.yaml
"""

import argparse
import glob
import logging
import os

import zarr

from fronts import callbacks as fronts_callbacks
from fronts import evaluate as fronts_evaluate
from fronts import model as fronts_model
from fronts import train as fronts_train
from fronts import utils
from fronts.data import datasets
from fronts.data import generate as era5_generate

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments and run pipeline stages in order, skipping completed ones."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run the full fronts pipeline: generate → train → eval")
    parser.add_argument("--config", type=str, required=True, help="Path to unified pipeline YAML config")
    args = parser.parse_args()

    yaml_data = utils.load_yaml(args.config)
    data_cfg = utils.parse_config_section(yaml_data, datasets.DatasetConfig, "data_config")
    model_cfg = utils.parse_config_section(yaml_data, fronts_model.ModelConfig, "model_config")
    callbacks_cfg = utils.parse_config_section(yaml_data, fronts_callbacks.CallbacksConfig, "callbacks_config")
    wandb_cfg = (
        utils.parse_config_section(yaml_data, fronts_train.WandBConfig, "wandb_config")
        if "wandb_config" in yaml_data
        else None
    )
    train_cfg = utils.parse_config_section(yaml_data, fronts_train.TrainConfig, "train_config")

    if "generate_config" in yaml_data and "icechunk_storage_config" in yaml_data:
        era5_cfg = utils.parse_config_section(
            yaml_data, era5_generate.ERA5DataLoaderConfig, "generate_config", utils.YAML_TYPE_HOOKS
        )
        icechunk_cfg = utils.parse_config_section(yaml_data, utils.IcechunkStorageConfig, "icechunk_storage_config")
        zarr.config.set({"async.concurrency": era5_cfg.zarr_async_concurrency})
        contents = era5_generate.inspect_store(icechunk_cfg)
        strategy = era5_generate.determine_write_strategy(era5_cfg, contents)
        if strategy.error_reason:
            raise ValueError(strategy.error_reason)
        if strategy.skip_reason:
            logger.info("Generate: skipping — %s", strategy.skip_reason)
        else:
            logger.info("Generate: starting ERA5 data generation...")
            strategy.execute(era5_cfg, icechunk_cfg)
            logger.info("Generate: complete.")

    if os.path.exists(f"{callbacks_cfg.model_checkpoint_path}_best_loss.keras"):
        logger.info("Train: skipping — checkpoint already exists at %s.", callbacks_cfg.model_checkpoint_path)
    else:
        logger.info("Train: starting...")
        fronts_train.train(data_cfg, model_cfg, callbacks_cfg, wandb_cfg, train_cfg)
        logger.info("Train: complete.")

    if "eval_config" in yaml_data:
        eval_cfg = utils.parse_config_section(
            yaml_data, fronts_evaluate.EvalConfig, "eval_config", utils.YAML_TYPE_HOOKS
        )
        if glob.glob(os.path.join(eval_cfg.outdir, "stats_aggregate*.nc")):
            logger.info("Eval: skipping — stats files already exist in %s.", eval_cfg.outdir)
        else:
            logger.info("Eval: starting...")
            fronts_evaluate.run(eval_cfg, data_cfg)
            logger.info("Eval: complete.")


if __name__ == "__main__":
    main()
