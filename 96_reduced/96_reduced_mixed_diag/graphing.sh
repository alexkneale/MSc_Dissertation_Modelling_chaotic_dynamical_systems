#!/bin/sh
# Grid Engine options (lines prefixed with #$)

# Name of job
#$ -N 96_red_g_mix_low
# Directory in which to run code (-cwd or -wd <path-to-wd>)
#$ -cwd
# Requested runtime allowance
# Requested runtime allowance
#$ -l h_rt=03:00:00
#$ -l h_vmem=240G
#$ -q gpu 
#$ -pe gpu-a100 1
# Email address for notifications
#$ -M s2028033@ed.ac.uk
# Option to request resource reservation
#$ -R y

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load conda
module load anaconda
# Activate conda environment
source activate surrogate_graphing

# Run the program
python graphing_96.py

