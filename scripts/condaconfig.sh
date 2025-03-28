#!/bin/bash

# Sets up HuggingFace Cache in Scratch
mkdir -p /storage/scratch1/4/akranias3/.cache/huggingface
export HF_HOME="/storage/scratch1/4/akranias3/.cache/huggingface"

# Sets conda install location to scratch
mkdir -p /storage/scratch1/4/akranias3/.conda/envs
mkdir -p /storage/scratch1/4/akranias3/.conda/pkgs

# Load SLURM Modules
module load anaconda3

# Configure Conda to use research group storage
conda config --add envs_dirs /storage/scratch1/4/akranias3/.conda/envs
conda config --add pkgs_dirs /storage/scratch1/4/akranias3/.conda/pkgs