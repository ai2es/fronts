#!/usr/bin/bash

#SBATCH --job-name=eval_ff_regions
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=mandelbaum.taylor@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=388G
#SBATCH -p ai2es
#SBATCH --output=/ourdisk/hpc/ai2es/tman/logs/eval_ff_regions_%j_out
#SBATCH --error=/ourdisk/hpc/ai2es/tman/logs/eval_ff_regions_%j_err
#SBATCH -t 48:00:00
#SBATCH --chdir=/home/tman/fronts

# Per-region stats + performance diagrams (full, land, ocean, and the five forecast-office
# regions) for the 2.0 checkpoint named in configs/sooner_eval_regions.yaml.

set -euo pipefail
export LD_LIBRARY_PATH="${PIXI_ENV}/lib:${LD_LIBRARY_PATH}"

CONFIG=configs/sooner_eval_regions.yaml
STATS_DIR=/ourdisk/hpc/ai2es/tman/models/stats/nbs_square_dilation_trof_tt_no_inst/

# One inference pass, then stats for every region in harness_eval_config.regions
pixi run -e schooner python -m fronts.evaluation.region_eval --config_path "$CONFIG"

# 4-panel performance diagrams for every stats_derived*.nc in the directory
pixi run -e schooner python -m fronts.evaluation.figures \
    --stats_dir "$STATS_DIR" \
    --coordinates 0.25 80.0 130.0 369.75
