#!/bin/bash
#SBATCH --job-name=train_yolov5_basic
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="b.a.companjen@library.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=5G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gpus=1

# load modules (assuming you start from the default environment)
# we explicitely call the modules to improve reproducability
# in case the default settings change
module load torchvision/0.8.2-fosscuda-2020b-PyTorch-1.7.1

source $HOME/venvs/yolo_env/bin/activate

echo "[$SHELL] #### Starting GPU YOLOv5 training"
echo "[$SHELL] This is $SLURM_JOB_USER and my job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] CWD: "$CWD

# Which GPU has been assigned
echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES


# Create a directory of local scratch on the node
echo "[$SHELL] Node scratch: "$SCRATCH
export RUNDIR=$SCRATCH/yolov5
cp -r $HOME/yolov5 $SCRATCH/
echo "[$SHELL] Run directory: "$RUNDIR

# Training configuration
export DATAYAML=$CWD/symbols.yaml

# Directory for training results is created by the script
export PROJECTDIR=$SCRATCH/runs

# Directory for training data must match $DATAYAML!
export DATASETS=$SCRATCH/datasets
mkdir -p $DATASETS
cp -r $HOME/mustafa $DATASETS/

# Change to $RUNDIR
cd $RUNDIR

# Run the file
echo "[$SHELL] Run script"
python train.py --data $DATAYAML --project $PROJECTDIR --weights yolov5m.pt --device $CUDA_VISIBLE_DEVICES --epochs 10
echo "[$SHELL] Script finished"

# Move stat directory back to CWD
echo "[$SHELL] Copy files back to cwd"
cp -r $PROJECTDIR $CWD/

echo "[$SHELL] #### Finished GPU training."
