#!/bin/bash
#SBATCH --job-name=test_gpu_tensorflow
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="b.a.companjen@library.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=5G
#SBATCH --time=00:02:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gpus=1

# load modules (assuming you start from the default environment)
# we explicitely call the modules to improve reproducability
# in case the default settings change
module load Python/3.7.4-GCCcore-8.3.0
module load SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-foss-2019b-Python-3.7.4
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4

echo "[$SHELL] #### Starting GPU TensorFlow test"
echo "[$SHELL] This is $SLURM_JOB_USER and my first job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] CWD: "$CWD

# Which GPU has been assigned
echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES

# Set the path to the python file
export PATH_TO_PYFILE=$CWD
echo "[$SHELL] Path of python file: "$PATH_TO_PYFILE

# Set name of the python file
export PYFILE=$CWD/test_python_gpu.py

# Create a directory of local scratch on the node
echo "[$SHELL] Node scratch: "$SCRATCH
export RUNDIR=$SCRATCH/test_tf
mkdir $RUNDIR
echo "[$SHELL] Run directory"$RUNDIR

# Create directory for plots
export PLOTDIR=$RUNDIR/plots
mkdir $PLOTDIR

# copy script to local scratch directory and change into it
cp $PYFILE $RUNDIR/
cd $RUNDIR

# Run the file
echo "[$SHELL] Run script"
python3 test_python_gpu.py
echo "[$SHELL] Script finished"

# Move stat directory back to CWD
echo "[$SHELL] Copy files back to cwd"
cp -r $PLOTDIR $CWD/

echo "[$SHELL] #### Finished GPU TensorFLow test. Have a nice day"
