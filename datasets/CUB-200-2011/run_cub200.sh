#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=cub200
#SBATCH --array=0-99
#SBATCH --output=cub200_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com
#SBATCH --exclude=g3070

source ~/.bashrc
conda activate /gscratch/krishna/kaityc/conda_envs/species

cd /mmfs1/gscratch/krishna/kaityc/SpeciesDataset/datasets/CUB-200-2011
python run.py
