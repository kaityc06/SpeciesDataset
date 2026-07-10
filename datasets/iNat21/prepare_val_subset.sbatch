#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=inat21_val_prep
#SBATCH --output=/gscratch/krishna/suryadv/SpeciesDataset_iNat21/logs/inat21_val_prep_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com
#SBATCH --exclude=g3110

set -eo pipefail

source /gscratch/krishna/kaityc/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/krishna/kaityc/conda_envs/species

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SCRIPT_DIR}"

export INAT21_RUNTIME_ROOT="${INAT21_RUNTIME_ROOT:-/gscratch/krishna/suryadv/SpeciesDataset_iNat21}"
export INAT21_DATA_DIR="${INAT21_DATA_DIR:-${INAT21_RUNTIME_ROOT}/data}"
export INAT21_OUTPUT_DIR="${INAT21_OUTPUT_DIR:-${INAT21_RUNTIME_ROOT}/outputs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${INAT21_RUNTIME_ROOT}/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export SPLIT="${SPLIT:-val}"
export INAT21_VAL_SUBSET_SIZE="${INAT21_VAL_SUBSET_SIZE:-2500}"
export INAT21_VAL_SUBSET_SEED="${INAT21_VAL_SUBSET_SEED:-42}"
export INAT21_VAL_SUBSET_MANIFEST="${INAT21_VAL_SUBSET_MANIFEST:-${INAT21_DATA_DIR}/val_species_balanced_subset_2500_seed42.json}"

mkdir -p "${INAT21_DATA_DIR}" "${INAT21_OUTPUT_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

python prepare_val_subset.py \
  --split "${SPLIT}" \
  --target-size "${INAT21_VAL_SUBSET_SIZE}" \
  --seed "${INAT21_VAL_SUBSET_SEED}" \
  --data-dir "${INAT21_DATA_DIR}" \
  --manifest "${INAT21_VAL_SUBSET_MANIFEST}"
