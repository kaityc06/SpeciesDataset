#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=fungi-download
#SBATCH --output=fungi_download_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com

source ~/.bashrc
conda activate /gscratch/krishna/kaityc/conda_envs/species

DATA_DIR="/mmfs1/gscratch/krishna/kaityc/SpeciesDataset/datasets/FGVCx-Fungi-2018/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[$(date)] Downloading FGVCx Fungi 2018 from Kaggle..."
kaggle competitions download -c fungi-challenge-fgvc-2018

echo "[$(date)] Extracting..."
unzip -o fungi-challenge-fgvc-2018.zip
rm fungi-challenge-fgvc-2018.zip

echo "[$(date)] Done. Contents:"
ls -lh "$DATA_DIR"
