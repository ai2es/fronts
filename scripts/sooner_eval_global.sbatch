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

# Global stats (global, NH, SH) for the 2.0 checkpoint named in configs/sooner_eval_global.yaml against
# the TWC global analysis, for direct, hemisphere-flipped and composite inference. Rasterize the GML
# first: python scripts/gml_to_zarr.py --gml_dir <extracted tar> --out <targets_zarr from the config>

set -eo pipefail
export LD_LIBRARY_PATH="${PIXI_ENV}/lib:${LD_LIBRARY_PATH}"

pixi run -e schooner python -m fronts.evaluation.global_eval --config_path configs/sooner_eval_global.yaml
