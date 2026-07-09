#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=inat21_val_prep
#SBATCH --output=inat21_val_prep_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com
#SBATCH --exclude=g3110

set -eo pipefail

source /gscratch/krishna/kaityc/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/krishna/kaityc/conda_envs/species

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export SPLIT="${SPLIT:-val}"
export INAT21_VAL_SUBSET_SIZE="${INAT21_VAL_SUBSET_SIZE:-2500}"
export INAT21_VAL_SUBSET_SEED="${INAT21_VAL_SUBSET_SEED:-42}"

python prepare_val_subset.py \
  --split "${SPLIT}" \
  --target-size "${INAT21_VAL_SUBSET_SIZE}" \
  --seed "${INAT21_VAL_SUBSET_SEED}"
