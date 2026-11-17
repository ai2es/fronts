#!/usr/bin/bash

#SBATCH --job-name=eval_ff_global_smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=mandelbaum.taylor@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=388G
#SBATCH -p ai2es
#SBATCH --output=/ourdisk/hpc/ai2es/tman/logs/eval_ff_global_smoke_%j_out
#SBATCH --error=/ourdisk/hpc/ai2es/tman/logs/eval_ff_global_smoke_%j_err
#SBATCH -t 02:00:00
#SBATCH --chdir=/home/tman/fronts

# One-week smoke test of the global eval (about 28 six-hourly timesteps, direct + flipped) before the
# full sooner_eval_global.sbatch run. Writes to a separate *_global_smoke/ directory.

set -eo pipefail
export LD_LIBRARY_PATH="${PIXI_ENV}/lib:${LD_LIBRARY_PATH}"

SMOKE_CONFIG=/ourdisk/hpc/ai2es/tman/logs/eval_global_smoke_${SLURM_JOB_ID}.yaml
sed 's|time_end: "2026-01-01|time_end: "2023-06-22|; s|_global/"|_global_smoke/"|' \
    configs/sooner_eval_global.yaml > "$SMOKE_CONFIG"
grep -E "time_end|outdir" "$SMOKE_CONFIG"

pixi run -e schooner python -m fronts.evaluation.global_eval --config_path "$SMOKE_CONFIG"
