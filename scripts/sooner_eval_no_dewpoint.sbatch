#!/usr/bin/bash

#SBATCH --job-name=eval_ff_nodew
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=mandelbaum.taylor@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=388G
#SBATCH -p ai2es
#SBATCH --output=/ourdisk/hpc/ai2es/tman/logs/eval_ff_nodew_%j_out
#SBATCH --error=/ourdisk/hpc/ai2es/tman/logs/eval_ff_nodew_%j_err
#SBATCH -t 48:00:00
#SBATCH --chdir=/home/tman/fronts

# USAD and global TWC evals for the nbs_square_dilation_trof_tt_no_inst_no_dewpoint run, the v2.0.1 model
# (v2.0.1.keras), which takes 27 input channels: the 2.0.0 stack without the 2 m dewpoint. The USAD config is
# committed as configs/sooner_eval_regions_v2.0.1.yaml. The global config is not committed, because it names where
# the rasterized TWC analyses live on Sooner, so it is derived from configs/sooner_eval_global.yaml here: run_name
# swapped and the 2m_dewpoint_temperature input dropped.
# The USAD stats and 4-panel figures (full, land, ocean and each forecast-office region) are written before the
# global eval starts. The global eval writes only
# the composite (NH from the direct pass, SH from the flipped one), and its 4-panel figures follow it.
# Global needs the rasterized GML at global_eval_config.targets_zarr (scripts/gml_to_zarr.py).

set -eo pipefail
export LD_LIBRARY_PATH="${PIXI_ENV}/lib:${LD_LIBRARY_PATH}"

RUN=nbs_square_dilation_trof_tt_no_inst_no_dewpoint
USAD_CFG=configs/sooner_eval_regions_v2.0.1.yaml
GLOBAL_CFG=/ourdisk/hpc/ai2es/tman/logs/sooner_eval_global_${RUN}_${SLURM_JOB_ID}.yaml
sed "s|^run_name: .*|run_name: ${RUN}|; /\"2m_dewpoint_temperature\",/d" \
    configs/sooner_eval_global.yaml > "$GLOBAL_CFG"
grep -c "2m_dewpoint_temperature" "$USAD_CFG" "$GLOBAL_CFG" || true  # expect 0 for both

STATS_ROOT=/ourdisk/hpc/ai2es/tman/models/stats
USAD_STATS=${STATS_ROOT}/${RUN}
GLOBAL_STATS=${STATS_ROOT}/${RUN}_global

# Each stage is skipped when its last output file exists, so a resubmit resumes where the last job
# stopped. Delete a stage's output to force a rerun.
stage() {
    local marker="$1"
    shift
    if [ -f "$marker" ]; then
        echo "==== skipping $* (found $marker)"
    else
        echo "==== $(date -u +%FT%TZ) :: $*"
        pixi run -e schooner "$@"
    fi
}

# USAD first, with its figures, so they exist even if the global eval runs out of time.
stage "$USAD_STATS/stats_derived_NHC.nc" python -m fronts.evaluation.region_eval --config_path "$USAD_CFG"
stage "$USAD_STATS/performance_INST_ocean.png" python -m fronts.evaluation.figures \
    --stats_dir "$USAD_STATS/" --coordinates 0.25 80.0 130.0 369.75
stage "$GLOBAL_STATS/composite/stats_derived_SH.nc" python -m fronts.evaluation.global_eval --config_path "$GLOBAL_CFG"
stage "$GLOBAL_STATS/composite/performance_TROF_SH.png" python -m fronts.evaluation.figures \
    --stats_dir "$GLOBAL_STATS/composite/" --coordinates -80.0 80.0 0.0 359.75
