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
#SBATCH --output=/gscratch/krishna/suryadv/SpeciesDataset_iNat21/logs/inat21_val_subset_%A_%a.log
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
export SPLIT="${SPLIT:-val}"
export MODE="${MODE:-process}"
export TOTAL_SHARDS="${TOTAL_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-10}}"
export NUM_SAMPLES="${NUM_SAMPLES:-250}"
export INAT21_VAL_SUBSET_SIZE="${INAT21_VAL_SUBSET_SIZE:-2500}"
export INAT21_VAL_SUBSET_SEED="${INAT21_VAL_SUBSET_SEED:-42}"
export INAT21_VAL_SUBSET_MANIFEST="${INAT21_VAL_SUBSET_MANIFEST:-${INAT21_DATA_DIR}/val_species_balanced_subset_2500_seed42.json}"
export SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-/gscratch/krishna/kaityc/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt}"
export HF_HOME="${HF_HOME:-${INAT21_RUNTIME_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/gscratch/krishna/kaityc/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${INAT21_RUNTIME_ROOT}/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export QWEN_HF_DOWNLOAD_LOCK="${QWEN_HF_DOWNLOAD_LOCK:-${INAT21_RUNTIME_ROOT}/cache/qwen_hf_download.lock}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="/gscratch/krishna/kaityc/SpeciesDataset:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/gscratch/krishna/kaityc/conda_envs/species/lib:${LD_LIBRARY_PATH:-}"

mkdir -p \
  "${INAT21_DATA_DIR}" \
  "${INAT21_OUTPUT_DIR}" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}" \
  "$(dirname "${QWEN_HF_DOWNLOAD_LOCK}")"

python run.py \
  --mode "${MODE}" \
  --num-samples "${NUM_SAMPLES}" \
  --shard-id "${SLURM_ARRAY_TASK_ID:-0}" \
  --total-shards "${TOTAL_SHARDS}"
