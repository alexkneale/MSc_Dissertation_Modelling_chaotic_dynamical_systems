#!/bin/bash
#
#SBATCH --job-name=t_96r_m_l
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=sc130

# Load the required modules

source ${HOME/home/work}/bin/activate

module load nvidia/nvhpc/22.11

python training.py

