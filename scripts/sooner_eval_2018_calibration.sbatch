#!/usr/bin/bash

#SBATCH --job-name=eval_2018_calibration
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=mandelbaum.taylor@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=388G
#SBATCH -p ai2es
#SBATCH --output=/ourdisk/hpc/ai2es/tman/logs/eval_2018_calibration_%j_out
#SBATCH --error=/ourdisk/hpc/ai2es/tman/logs/eval_2018_calibration_%j_err
#SBATCH -t 48:00:00
#SBATCH --chdir=/home/tman/fronts

# 2018 (validation year) stats for the v2.0.0 checkpoint, the neighborhood calibration table fit
# from them, and the 4-panel performance diagrams. The checkout in --chdir must include
# fronts.neighborhood_calibration: branch feature/neighborhood-calibration, or master once it merges.

set -eo pipefail
export LD_LIBRARY_PATH="${PIXI_ENV}/lib:${LD_LIBRARY_PATH}"

CONFIG=configs/sooner_eval_2018.yaml
STATS_DIR=/ourdisk/hpc/ai2es/tman/models/stats/nbs_square_dilation_trof_tt_no_inst_2018  # eval_config.outdir in $CONFIG

# Stop in seconds, not after the eval, if this checkout predates the calibration code.
pixi run -e schooner python -c "import fronts.neighborhood_calibration"

# Full-domain stats for 2018, including the per-radius truth counts the calibration needs
pixi run -e schooner python -m fronts.evaluate --config_path "$CONFIG"

# Raw probability -> chance of a front within 50-250 km, for each front type
pixi run -e schooner python -m fronts.neighborhood_calibration \
    --stats_path "$STATS_DIR/stats_aggregate.nc" \
    --output_path "$STATS_DIR/neighborhood_calibration.nc"

# 4-panel performance diagrams; the 0 km curve shows calibration at the pixel scale
pixi run -e schooner python -m fronts.evaluation.figures \
    --stats_dir "$STATS_DIR" \
    --coordinates 0.25 80.0 130.0 369.75
