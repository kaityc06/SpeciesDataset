#!/bin/bash
# Snapshot Serengeti-only submit wrapper.
#
# This mirrors the resource/runtime setup from scripts/submit_process.sbatch,
# but calls datasets/SnapshotSerengeti/run_stable_process.py so the workaround
# stays local to Snapshot Serengeti. The shared scripts remain unchanged.
#
# The job emails Surya when an array task ends or fails.

#SBATCH --account=krishna
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=SnapshotSerengeti-stable
#SBATCH --output=/gscratch/scrubbed/suryadv/repos/SpeciesDataset/logs/process_%x_%A_%a.log
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

TOTAL_SHARDS="${TOTAL_SHARDS:-400}"
NUM_SAMPLES="${NUM_SAMPLES:-250}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"

export SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-/gscratch/krishna/kaityc/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt}"
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/repos/SpeciesDataset/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/gscratch/scrubbed/suryadv/repos/SpeciesDataset/.hf_cache/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/suryadv/repos/SpeciesDataset/.cache}"
export PYTHONPATH="/gscratch/scrubbed/suryadv/repos/sam3-testing:${PYTHONPATH:-}"

SPECIES_ENV="/gscratch/krishna/kaityc/conda_envs/species"
if command -v conda >/dev/null 2>&1; then
  source ~/.bashrc
  conda activate "${SPECIES_ENV}"
  PYTHON_BIN="python"
else
  PYTHON_BIN="${SPECIES_ENV}/bin/python"
fi

# ckpt nodes may expose an older system libstdc++. Keep the conda runtime
# libraries first so PIL/libLerc can find GLIBCXX_3.4.29.
export LD_LIBRARY_PATH="${SPECIES_ENV}/lib:${LD_LIBRARY_PATH:-}"

cd /gscratch/scrubbed/suryadv/repos/SpeciesDataset
mkdir -p logs "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}"

echo "[snapshot_stable_submit] shard=${SHARD_ID}/${TOTAL_SHARDS} num_samples=${NUM_SAMPLES}"
echo "[snapshot_stable_submit] output_subdir=${SPECIES_OUTPUT_SUBDIR:-<root outputs>}"
echo "[snapshot_stable_submit] sam3_checkpoint=${SAM3_CHECKPOINT_PATH}"
echo "[snapshot_stable_submit] hf_home=${HF_HOME}"
echo "[snapshot_stable_submit] hf_datasets_cache=${HF_DATASETS_CACHE}"

"${PYTHON_BIN}" "datasets/SnapshotSerengeti/run_stable_process.py" \
  --num-samples "${NUM_SAMPLES}" \
  --shard-id "${SHARD_ID}" \
  --total-shards "${TOTAL_SHARDS}"
