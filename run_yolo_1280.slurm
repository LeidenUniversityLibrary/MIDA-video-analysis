#!/bin/bash
#SBATCH --job-name=run_yolov5
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="b.a.companjen@library.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=15G
#SBATCH --time=01:30:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gpus=1

# load modules (assuming you start from the default environment)
# we explicitely call the modules to improve reproducability
# in case the default settings change
module load torchvision/0.8.2-fosscuda-2020b-PyTorch-1.7.1

source $HOME/venvs/yolo_env/bin/activate

echo "[$SHELL] #### Starting GPU YOLOv5 detection"
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

# Directory for detection results is created by the script
export PROJECTDIR=$SCRATCH/detection

export MODEL=$SCRATCH/yolo_1280.pt
cp $HOME/models/yolo_1280.pt $MODEL
# Directory for datasets
export DATASETS=$SCRATCH/datasets
mkdir -p $DATASETS
# Source to run detection on
cp -r $HOME/frames_by_class.tar $HOME/star.zip $DATASETS/
cd $DATASETS
tar -xf frames_by_class.tar
unzip star.zip

# Change to $RUNDIR
cd $RUNDIR

# Run the file
echo "[$SHELL] Run script"
python detect.py --source $DATASETS/'star_of_david/*.jpg' --project $PROJECTDIR --name star --img 1280 --weights $MODEL --device $CUDA_VISIBLE_DEVICES --save-txt --save-conf
python detect.py --source $DATASETS/'no_star_of_david/*.jpg' --project $PROJECTDIR --name no_star --img 1280 --weights $MODEL --device $CUDA_VISIBLE_DEVICES --save-txt --save-conf
python detect.py --source $DATASETS/'frames_by_class/pentagram/**/*.jpg' --project $PROJECTDIR --name pentagram --img 1280 --weights $MODEL --device $CUDA_VISIBLE_DEVICES --save-txt --save-conf
python detect.py --source $DATASETS/'frames_by_class/no_pentagram/**/*.jpg' --project $PROJECTDIR --name no_pentagram --img 1280 --weights $MODEL --device $CUDA_VISIBLE_DEVICES --save-txt --save-conf
echo "[$SHELL] Script finished"

# Move stat directory back to CWD
echo "[$SHELL] Remove dataset of images"
rm -rf $DATASETS
echo "[$SHELL] Zip and copy files back to cwd"
cd $PROJECTDIR
zip -r $SCRATCH/results_$SLURM_JOB_ID.zip ./ 
cp -r $SCRATCH/results_$SLURM_JOB_ID.zip $CWD/

echo "[$SHELL] #### Finished detection."
