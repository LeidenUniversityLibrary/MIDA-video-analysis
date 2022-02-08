#!/bin/bash
#SBATCH --job-name=test_python_simple
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="b.a.companjen@library.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=100M
#SBATCH --time=00:01:00
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
module load SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python test_python_simple.py
echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"