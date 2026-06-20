#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=deepweeds-download
#SBATCH --output=deepweeds_download_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com

source ~/.bashrc
conda activate /gscratch/krishna/kaityc/conda_envs/species

DATA_DIR="/mmfs1/gscratch/krishna/kaityc/SpeciesDataset/datasets/DeepWeeds/data"
IMAGES_DIR="$DATA_DIR/images"
mkdir -p "$IMAGES_DIR"

echo "[$(date)] Downloading DeepWeeds images (~468 MB) from Google Drive..."
cd "$DATA_DIR"
gdown --no-cookies 1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj -O deepweeds.zip

echo "[$(date)] Extracting..."
unzip -q deepweeds.zip -d "$IMAGES_DIR"
rm deepweeds.zip

echo "[$(date)] Downloading labels CSV..."
wget -q https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/labels.csv -O "$DATA_DIR/labels.csv"

echo "[$(date)] Done. Image count: $(ls "$IMAGES_DIR" | wc -l)"
