#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --job-name=wildlifereid-download
#SBATCH --output=wildlifereid_download_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com

source ~/.bashrc
conda activate /gscratch/krishna/kaityc/conda_envs/species

DATA_DIR="/mmfs1/gscratch/krishna/kaityc/SpeciesDataset/datasets/WildlifeReID-10k/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[$(date)] Downloading WildlifeReID-10k from Kaggle (~23 GB)..."
kaggle datasets download -d wildlifedatasets/wildlifereid-10k

echo "[$(date)] Extracting..."
unzip -o wildlifereid-10k.zip
rm wildlifereid-10k.zip

echo "[$(date)] Done. Contents:"
ls -lh "$DATA_DIR"
