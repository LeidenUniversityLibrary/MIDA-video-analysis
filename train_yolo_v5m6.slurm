#!/bin/bash
#SBATCH --job-name=t_m4_yolov5
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="b.a.companjen@library.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=25G
#SBATCH --time=30:00:00
#SBATCH --partition=gpu-long
#SBATCH --ntasks=1
#SBATCH --gpus=1

# load modules (assuming you start from the default environment)
# we explicitely call the modules to improve reproducability
# in case the default settings change
module load torchvision/0.8.2-fosscuda-2020b-PyTorch-1.7.1

# Ben set up a virtual environment with packages required by yolov5,
# that are not loaded by the torchvision module
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
export DATAYAML=$CWD/symbols2.yaml

# Directory for training results is created by the script
export PROJECTDIR=$SCRATCH/training
export PROJECTNAME=yolo_model_4

# Directory for training data must match $DATAYAML!
export DATASETS=$SCRATCH/datasets
mkdir -p $DATASETS
cp -r $HOME/training_data $DATASETS/

# Change to $RUNDIR
cd $RUNDIR

# Run the file
echo "[$SHELL] Run script"
python train.py --data $DATAYAML --project $PROJECTDIR --name $PROJECTNAME --weights yolov5m6.pt --batch 4 --img 1280 --device $CUDA_VISIBLE_DEVICES --epochs 400
echo "[$SHELL] Script finished"

# Move stat directory back to CWD
echo "[$SHELL] Copy files back to cwd"
cd $PROJECTDIR
zip -r $SCRATCH/$PROJECTNAME-$SLURM_JOB_ID.zip ./ 
cp -r $SCRATCH/$PROJECTNAME-$SLURM_JOB_ID.zip $CWD/

echo "[$SHELL] #### Finished GPU training."
