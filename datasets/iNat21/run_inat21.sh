#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=inat21
#SBATCH --array=0-99
#SBATCH --output=inat21_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com
#SBATCH --exclude=g3110

source ~/.bashrc
conda activate /gscratch/krishna/kaityc/conda_envs/species

cd /mmfs1/gscratch/krishna/kaityc/SpeciesDataset/datasets/iNat21

# Override split at submission time: SPLIT=val sbatch run_inat21.sh
SPLIT=${SPLIT:-train} python run.py
