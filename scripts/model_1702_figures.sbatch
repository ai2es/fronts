#!/bin/bash
#SBATCH --job-name=model_1702_figures
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/ourdisk/hpc/ai2es/tman/logs/model_1702_figures_%j_out.txt
#SBATCH --error=/ourdisk/hpc/ai2es/tman/logs/model_1702_figures_%j_err.txt

# Renders the 4-panel performance figures from already-computed model_1702/baseline
# stats_derived*.nc files (stage 3 of scripts/model_1702_comparison.sbatch), without
# re-running side-store generation, evaluation, or the cross-model comparison summary.
# Submit from the repo root on branch harness/model_1702-comparison:
#
#     sbatch scripts/model_1702_figures.sbatch
#
# No GPU needed — this only reads stats NetCDFs and plots them.

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
STATS_ROOT="/ourdisk/hpc/ai2es/tman/models/stats/model_1702"
cd "$REPO_DIR"

run() {
    echo "==== $(date -u +%FT%TZ) :: $*"
    pixi run -e schooner "$@"
}

echo "==== figures ===="
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

echo "==== $(date -u +%FT%TZ) :: figures under $STATS_ROOT"
