#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --job-name=iwildcam-extract
#SBATCH --output=outputs/iwildcam_extract_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com

TAR_PATH="/mmfs1/gscratch/krishna/sgeng/datasets/iwildcam.tar"
DATA_DIR="/mmfs1/gscratch/krishna/kaityc/SpeciesDataset/datasets/iWildCam/data"

mkdir -p "$DATA_DIR"

echo "[$(date)] Extracting iWildCam v2.0 (~12 GB) from $TAR_PATH ..."
tar -xf "$TAR_PATH" --strip-components=1 -C "$DATA_DIR"

echo "[$(date)] Done. Contents:"
ls -lh "$DATA_DIR"
echo "Train images: $(ls "$DATA_DIR/train/" | wc -l)"
