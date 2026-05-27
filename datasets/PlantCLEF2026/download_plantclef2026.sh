#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=plantclef2026_download
#SBATCH --output=plantclef2026_download_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com

BASE_URL="https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data"
DEST="/mmfs1/gscratch/krishna/kaityc/lifeclef2026_cache/plantclef2026"

mkdir -p "$DEST"
cd "$DEST"

echo "=== Downloading training metadata CSV ==="
wget -c --progress=dot:mega \
    "${BASE_URL}/PlantCLEF2024singleplanttrainingdata.csv" \
    -O PlantCLEF2024singleplanttrainingdata.csv

echo "=== Downloading training images (800 px max side, ~160 GB) ==="
wget -c --progress=dot:giga \
    "${BASE_URL}/PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar" \
    -O PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar

echo "=== Extracting images ==="
tar -xf PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar

echo "=== Done. Files in $DEST ==="
ls -lh "$DEST"
