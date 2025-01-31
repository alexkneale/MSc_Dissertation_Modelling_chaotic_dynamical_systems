#!/bin/sh
# Grid Engine options (lines prefixed with #$)

# Name of job
#$ -N g_96_r_det
# Directory in which to run code (-cwd or -wd <path-to-wd>)
#$ -cwd
# Requested runtime allowance

#$ -l h_rt=00:05:00
# Requested memory (per core)
# Requested number of cores in parallel environment
#$ -l h_vmem=1G
# Requested number of cores in parallel environment
#$ -pe sharedmem 1
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
python graphing_loss.py

