#!/bin/bash
#
#SBATCH --job-name=d_f_m_l
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --account=sc130

# Load the required modules

source ${HOME/home/work}/bin/activate

module load nvidia/nvhpc/22.11

python data_collection.py

