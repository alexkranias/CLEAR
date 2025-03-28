#!/bin/bash

# Load SLURM Modules
module load anaconda3

# Setup Conda
conda create -n super_resolution_experiment python=3.9
eval "$(conda shell.bash hook)" # sets up conda in cli
conda activate super_resolution_experiment

# Install Dependencies
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt

export HF_HOME="/storage/scratch1/4/akranias3/.cache/huggingface"
huggingface-cli login --token $(cat ./scripts/token.txt)

# Run experiement
python experiment.py