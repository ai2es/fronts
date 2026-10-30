#!/bin/bash
#SBATCH --job-name=compare_confidence
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/ourdisk/hpc/ai2es/tman/logs/compare_confidence_%j_out.txt
#SBATCH --error=/ourdisk/hpc/ai2es/tman/logs/compare_confidence_%j_err.txt

# Measures whether a 2.0 run actually emits lower peak probabilities than model_1702,
# from the prediction arrays rather than from rendered figures. Submit from the repo root:
#
#     sbatch scripts/compare_confidence.sbatch
#
# Override the run under test without editing this file:
#
#     RUN_CONFIG=configs/sooner_eval_3d.yaml sbatch scripts/compare_confidence.sbatch
#     MAX_TIMESTEPS=32 sbatch scripts/compare_confidence.sbatch
#
# To point at a different training run, set `run_name` at the top of the eval YAML --
# model_path interpolates it as ${run_name}, so no path editing is needed.
#
# Both models are forced onto COORDINATES below so they score the same domain. That box is
# model_1702's CONUS training extent; sooner_eval.yaml otherwise defaults to the full USAD
# domain, which would make the two columns incomparable.

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

RUN_CONFIG="${RUN_CONFIG:-configs/sooner_eval.yaml}"
REFERENCE_CONFIG="${REFERENCE_CONFIG:-configs/model_1702/eval_1702_conus_6h.yaml}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-8}"
COORDINATES="${COORDINATES:-25.0 56.75 228.0 299.75}"

echo "==== $(date -u +%FT%TZ) :: comparing $RUN_CONFIG against $REFERENCE_CONFIG"
echo "==== domain: $COORDINATES | timesteps: $MAX_TIMESTEPS"

# shellcheck disable=SC2086
pixi run -e schooner python scripts/compare_confidence.py \
    --config "$REFERENCE_CONFIG" \
    --config "$RUN_CONFIG" \
    --max-timesteps "$MAX_TIMESTEPS" \
    --coordinates $COORDINATES

echo "==== $(date -u +%FT%TZ) :: done -- table is in the job's out.txt"
