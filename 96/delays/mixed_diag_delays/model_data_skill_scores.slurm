#!/bin/bash
#
#SBATCH --job-name=g_l_63_m_d
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=sc130

# Load the required modules

source ${HOME/home/work}/bin/activate

module load nvidia/nvhpc/22.11

python model_data_skill_scores.py

