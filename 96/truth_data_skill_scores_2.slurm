#!/bin/bash
#
#SBATCH --job-name=traj_63
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=sc130

# Load the required modules

source ${HOME/home/work}/bin/activate

module load nvidia/nvhpc/22.11

python truth_data_skill_scores_2.py

