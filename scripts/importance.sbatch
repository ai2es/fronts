#!/bin/bash
#SBATCH --job-name=importance
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --output=/ourdisk/hpc/ai2es/tman/logs/importance_%j_out.txt
#SBATCH --error=/ourdisk/hpc/ai2es/tman/logs/importance_%j_err.txt

# Permutation-importance ranking of input variables (src/fronts/importance.py).
# Submit from the repo root:
#
#     sbatch scripts/importance.sbatch configs/schooner_importance.yaml
#     sbatch scripts/importance.sbatch configs/schooner_importance.yaml land
#
# Runtime scales with n_repeats x len(variables) x one compute_stats pass (each pass
# costs about as much as one evaluate.py run), so raise --time above if the config
# covers many variables/repeats or a large time range.

set -euo pipefail

CONFIG_PATH="${1:?Usage: sbatch scripts/importance.sbatch <config_path> [mask]}"
MASK="${2:-}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

echo "==== $(date -u +%FT%TZ) :: importance --config_path $CONFIG_PATH ${MASK:+--mask $MASK}"
if [ -n "$MASK" ]; then
    pixi run -e schooner python src/fronts/importance.py --config_path "$CONFIG_PATH" --mask "$MASK"
else
    pixi run -e schooner python src/fronts/importance.py --config_path "$CONFIG_PATH"
fi

echo "==== $(date -u +%FT%TZ) :: importance complete"
