#!/bin/sh
# Grid Engine options (lines prefixed with #$)

# Name of job
#$ -N t_96_r_s_d
# Directory in which to run code (-cwd or -wd <path-to-wd>)
#$ -cwd
#$ -l rl9=true
# Requested runtime allowance
#$ -l h_rt=20:00:00

#$ -l h_vmem=240G
#$ -q gpu 
#$ -pe gpu-a100 1
#Email address for notifications
#$ -M s2028033@ed.ac.uk
# Option to request resource reservation
#$ -R y

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load conda
module load anaconda
# Activate conda environment
source activate surrogate_t

# Run the program
python training.py

