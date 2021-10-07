#!/bin/bash
#SBATCH --job-name=train_multihot_model
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

echo "[$SHELL] #### Starting GPU training job"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] CWD: "$CWD

# Which GPU has been assigned
echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES

# Set the path to the python file
export PATH_TO_PYFILE=$CWD
echo "[$SHELL] Path of python file: "$PATH_TO_PYFILE

# Set name of the python file
export SCRIPTNAME=build_multi_hot_model.py
export PYFILE=$CWD/$SCRIPTNAME
# Create a directory of local scratch on the node
echo "[$SHELL] Node scratch: "$SCRATCH
export RUNDIR=$SCRATCH

export IMAGE_DIR=$RUNDIR/images
export MODEL_NAME=multihotmodel_$SLURM_JOB_ID
export CSV_FILE=annotations.csv
mkdir -p $IMAGE_DIR

# Copy training data to scratch
cp images.tar $IMAGE_DIR/
tar -xf $IMAGE_DIR/images.tar
# copy script to local scratch directory and change into it
cp $PYFILE $RUNDIR/
cd $RUNDIR

# Run the file
echo "[$SHELL] Run script"
python3 $SCRIPTNAME $IMAGE_DIR $CSV_FILE $MODEL_NAME
echo "[$SHELL] Script finished"

# Move stat directory back to CWD
echo "[$SHELL] Copy files back to cwd"
cp -r $MODEL_NAME $CWD/

echo "[$SHELL] #### Finished training."
