#!/usr/bin/env bash

#SBATCH --job-name=model_run
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

#--batch-size 256 --deep True --noise True --reduced True --pixel True
python model.py --deep True --pixel True