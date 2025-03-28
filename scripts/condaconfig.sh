#!/bin/bash

mkdir -p /storage/scratch1/4/akranias3/.conda/envs
mkdir -p /storage/scratch1/4/akranias3/.conda/pkgs

# Load SLURM Modules
module load anaconda3

# Configure Conda to use research group storage
conda config --add envs_dirs /storage/scratch1/4/akranias3/.conda/envs
conda config --add pkgs_dirs /storage/scratch1/4/akranias3/.conda/pkgs