#!/bin/bash
#SBATCH --job-name=model_1702_comparison
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/ourdisk/hpc/ai2es/tman/logs/model_1702_comparison_%j_out.txt
#SBATCH --error=/ourdisk/hpc/ai2es/tman/logs/model_1702_comparison_%j_err.txt

# End-to-end model_1702 vs baseline comparison for 2019: side-store generation, all
# evaluations, 4-panel figures, and the cross-model summary. Submit from the repo root on
# branch harness/model_1702-comparison:
#
#     sbatch scripts/model_1702_comparison.sbatch
#
# Prerequisites (runbook steps 1-2): branch checked out on schooner, pixi -e schooner solved,
# model_1702.h5 at /ourdisk/hpc/ai2es/tman/models/model_1702/, and the baseline checkpoint at
# /ourdisk/hpc/ai2es/tman/models/brier_3d_full_domain_pretrained_conus/_best_loss.keras.
# Inputs come from the anonymous ARCO ERA5 store (Arraylake lacks the 950/900 hPa levels),
# so no Arraylake auth is needed.
#
# Resubmit-safe: side-store generation resumes from the last committed batch, and each eval
# stage is skipped once its final region's stats_derived file exists. Delete a stage's stats
# directory to force a rerun.

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
STATS_ROOT="/ourdisk/hpc/ai2es/tman/models/stats/model_1702"
cd "$REPO_DIR"

run() {
    echo "==== $(date -u +%FT%TZ) :: $*"
    pixi run -e schooner "$@"
}

eval_stage() {
    local config="$1"
    local marker="$2"
    if [ -f "$marker" ]; then
        echo "==== skipping $config (found $marker)"
    else
        run python -m fronts.model_1702.run_eval --config_path "$config"
    fi
}

echo "==== stage 1/4: side-store generation (resume-safe) ===="
run python -m fronts.model_1702.store --config configs/model_1702/generate_conus.yaml
run python -m fronts.model_1702.store --config configs/model_1702/generate_full.yaml

echo "==== stage 2/4: evaluations ===="
eval_stage configs/model_1702/eval_1702_conus.yaml    "$STATS_ROOT/model_1702_conus/stats_derived_ocean.nc"
eval_stage configs/model_1702/eval_1702_conus_6h.yaml "$STATS_ROOT/model_1702_conus_6h/stats_derived_ocean.nc"
eval_stage configs/model_1702/eval_baseline_conus.yaml "$STATS_ROOT/baseline_conus/stats_derived_ocean.nc"
eval_stage configs/model_1702/eval_1702_full.yaml     "$STATS_ROOT/model_1702_full_out_of_domain/stats_derived_NHC.nc"
eval_stage configs/model_1702/eval_baseline_full.yaml "$STATS_ROOT/baseline_full/stats_derived_NHC.nc"

echo "==== stage 3/4: figures ===="
for run_name in model_1702_conus model_1702_conus_6h baseline_conus; do
    run python -m fronts.model_1702.figures \
        --stats_dir "$STATS_ROOT/$run_name/" \
        --coordinates 25.0 56.75 228.0 299.75
done
for run_name in model_1702_full_out_of_domain baseline_full; do
    run python -m fronts.model_1702.figures \
        --stats_dir "$STATS_ROOT/$run_name/" \
        --coordinates 0.25 80.0 130.0 369.75
done

echo "==== stage 4/4: comparison summary ===="
run python -m fronts.model_1702.compare \
    --stats_root "$STATS_ROOT/" \
    --reference model_1702_conus_6h \
    --out "$STATS_ROOT/comparison.csv"

echo "==== $(date -u +%FT%TZ) :: comparison complete — stats, figures, and comparison.csv under $STATS_ROOT"
