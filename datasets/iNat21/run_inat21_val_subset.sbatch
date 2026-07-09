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

source /gscratch/krishna/kaityc/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/krishna/kaityc/conda_envs/species

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SCRIPT_DIR}"

export SPLIT="${SPLIT:-val}"
export MODE="${MODE:-process}"
export TOTAL_SHARDS="${TOTAL_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-10}}"
export NUM_SAMPLES="${NUM_SAMPLES:-250}"
export INAT21_VAL_SUBSET_SIZE="${INAT21_VAL_SUBSET_SIZE:-2500}"
export INAT21_VAL_SUBSET_SEED="${INAT21_VAL_SUBSET_SEED:-42}"
export SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-/gscratch/krishna/kaityc/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt}"
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/repos/SpeciesDataset/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/suryadv/repos/SpeciesDataset/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export PYTHONPATH="/gscratch/krishna/kaityc/SpeciesDataset:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/gscratch/krishna/kaityc/conda_envs/species/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

python run.py \
  --mode "${MODE}" \
  --num-samples "${NUM_SAMPLES}" \
  --shard-id "${SLURM_ARRAY_TASK_ID:-0}" \
  --total-shards "${TOTAL_SHARDS}"
