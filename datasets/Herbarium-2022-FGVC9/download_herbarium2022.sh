#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=herbarium2022-download
#SBATCH --output=herbarium2022_download_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com

source ~/.bashrc
conda activate /gscratch/krishna/kaityc/conda_envs/species

META_DIR="/mmfs1/gscratch/krishna/kaityc/SpeciesDataset/datasets/Herbarium-2022-FGVC9/data"
SCRUBBED_DIR="/gscratch/scrubbed/kaityc/herbarium2022"

mkdir -p "$META_DIR"
mkdir -p "$SCRUBBED_DIR"
cd "$SCRUBBED_DIR"

echo "[$(date)] Downloading Herbarium 2022 FGVC9 from Kaggle..."
kaggle competitions download -c herbarium-2022-fgvc9

echo "[$(date)] Extracting..."
unzip -o herbarium-2022-fgvc9.zip
rm herbarium-2022-fgvc9.zip

# Move metadata files to the persistent data dir
mv -f "$SCRUBBED_DIR/train_metadata.json" "$META_DIR/" 2>/dev/null || true
mv -f "$SCRUBBED_DIR/sample_submission.csv" "$META_DIR/" 2>/dev/null || true

# Symlink train_images into the data dir so config.py can find images
ln -sfn "$SCRUBBED_DIR/train_images" "$META_DIR/train_images"

echo "[$(date)] Done. Contents:"
ls -lh "$SCRUBBED_DIR"
ls -lh "$META_DIR"
