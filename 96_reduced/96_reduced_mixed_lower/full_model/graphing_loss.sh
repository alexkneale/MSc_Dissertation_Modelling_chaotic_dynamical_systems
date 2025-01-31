#!/bin/sh
# Grid Engine options (lines prefixed with #$)

# Name of job
#$ -N g_l_96_m_l
# Directory in which to run code (-cwd or -wd <path-to-wd>)
#$ -cwd
# Requested runtime allowance
#$ -l rl9=true
#$ -l h_rt=00:10:00
# Requested memory (per core)
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

