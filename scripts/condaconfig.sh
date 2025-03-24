#!/bin/bash

# Load SLURM Modules
module load anaconda3

# Configure Conda to use research group storage
conda config --add envs_dirs /storage/coda1/p-hshi308/0/akranias3/.conda/envs
conda config --add pkgs_dirs /storage/coda1/p-hshi308/0/akranias3/.conda/pkgs