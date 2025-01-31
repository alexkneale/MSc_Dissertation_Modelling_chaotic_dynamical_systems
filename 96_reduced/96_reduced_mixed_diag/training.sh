#!/bin/sh
# Grid Engine options (lines prefixed with #$)

# Name of job
#$ -N t_96_r_m_d
# Directory in which to run code (-cwd or -wd <path-to-wd>)
#$ -cwd
# Requested runtime allowance
#$ -l h_rt=33:00:00
#$ -l rl9=true
#$ -l h_vmem=6G
#$ -pe sharedmem 40
#Email address for notifications
#$ -M s2028033@ed.ac.uk
# Option to request resource reservation
#$ -R y

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load conda
module load anaconda
# Activate conda environment
source activate surrogate3

# Run the program
python training.py

