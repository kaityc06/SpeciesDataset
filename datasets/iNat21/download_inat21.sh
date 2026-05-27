#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --job-name=inat21_extract
#SBATCH --output=inat21_extract.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaitlynchen06@gmail.com

cd /mmfs1/gscratch/krishna/kaityc/inat21_cache
wget -c https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz
tar -xzf train.tar.gz
