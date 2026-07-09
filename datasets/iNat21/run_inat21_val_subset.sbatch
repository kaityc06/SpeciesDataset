#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=inat21_val_masks
#SBATCH --array=0-9
#SBATCH --output=inat21_val_subset_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com
#SBATCH --exclude=g3110

set -eo pipefail

source ~/.bashrc
conda activate /gscratch/krishna/kaityc/conda_envs/species

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export SPLIT="${SPLIT:-val}"
export MODE="${MODE:-process}"
export TOTAL_SHARDS="${TOTAL_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-10}}"
export NUM_SAMPLES="${NUM_SAMPLES:-250}"
export INAT21_VAL_SUBSET_SIZE="${INAT21_VAL_SUBSET_SIZE:-2500}"
export INAT21_VAL_SUBSET_SEED="${INAT21_VAL_SUBSET_SEED:-42}"

python run.py \
  --mode "${MODE}" \
  --num-samples "${NUM_SAMPLES}" \
  --shard-id "${SLURM_ARRAY_TASK_ID:-0}" \
  --total-shards "${TOTAL_SHARDS}"
