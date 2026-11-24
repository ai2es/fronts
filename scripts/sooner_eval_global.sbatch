#!/usr/bin/bash

#SBATCH --job-name=eval_ff_global
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=mandelbaum.taylor@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=388G
#SBATCH -p ai2es
#SBATCH --output=/ourdisk/hpc/ai2es/tman/logs/eval_ff_global_%j_out
#SBATCH --error=/ourdisk/hpc/ai2es/tman/logs/eval_ff_global_%j_err
#SBATCH -t 48:00:00
#SBATCH --chdir=/home/tman/fronts

# Global stats (global, NH, SH) for the 2.0 checkpoint named in your global eval config, scored against
# the TWC global analysis: the composite of direct (NH) and hemisphere-flipped (SH) inference.
#
# That config is not in the repository, because it names where the TWC analyses and their rasterized fronts
# live. Keep it in your own run directory and pass it in:
#   GLOBAL_CONFIG=/path/to/sooner_eval_global.yaml sbatch scripts/sooner_eval_global.sbatch
# Its sections are documented in src/fronts/evaluation/global_eval.py. Rasterize the GML first:
#   python scripts/gml_to_zarr.py --gml_dir <extracted tar> --out <the targets_zarr your config names>

set -eo pipefail
export LD_LIBRARY_PATH="${PIXI_ENV}/lib:${LD_LIBRARY_PATH}"

GLOBAL_CONFIG=${GLOBAL_CONFIG:?set GLOBAL_CONFIG to your global eval config; it is not committed}
pixi run -e schooner python -m fronts.evaluation.global_eval --config_path "$GLOBAL_CONFIG"
