#!/bin/bash
#
#SBATCH --job-name=gl_96r_m_l
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=00:02:00
#SBATCH --gres=gpu:1
#SBATCH --account=sc130

# Load the required modules

source ${HOME/home/work}/bin/activate

module load nvidia/nvhpc/22.11

python graphing_loss.py

