#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=herbarium2022
#SBATCH --array=0-399
#SBATCH --output=herbarium2022_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com

source ~/.bashrc
conda activate /gscratch/krishna/kaityc/conda_envs/species

cd /mmfs1/gscratch/krishna/kaityc/SpeciesDataset/datasets/Herbarium-2022-FGVC9
python run.py
