#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=cub200-download
#SBATCH --output=cub200_download_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com

source ~/.bashrc
conda activate /gscratch/krishna/kaityc/conda_envs/species

DATA_DIR="/mmfs1/gscratch/krishna/kaityc/SpeciesDataset/datasets/CUB-200-2011/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[$(date)] Downloading CUB-200-2011 (~1.1 GB)..."
wget -q --show-progress https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

echo "[$(date)] Extracting..."
tar -xzf CUB_200_2011.tgz
rm CUB_200_2011.tgz

echo "[$(date)] Done. Contents:"
ls -lh "$DATA_DIR/CUB_200_2011"
