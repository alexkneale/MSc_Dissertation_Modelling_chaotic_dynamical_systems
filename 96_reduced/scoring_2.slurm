#!/bin/bash
#
#SBATCH --job-name=data_96_r
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=sc130

# Load the required modules

source ${HOME/home/work}/bin/activate

module load nvidia/nvhpc/22.11

python scoring_2.py

