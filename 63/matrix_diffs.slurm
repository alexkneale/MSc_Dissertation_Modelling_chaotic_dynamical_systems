#!/bin/bash
#
#SBATCH --job-name=data_63
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=00:02:00
#SBATCH --gres=gpu:1
#SBATCH --account=sc130

# Load the required modules

source ${HOME/home/work}/bin/activate

module load nvidia/nvhpc/22.11

python matrix_diffs.py

