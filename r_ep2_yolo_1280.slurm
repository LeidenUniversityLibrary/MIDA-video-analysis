#!/bin/bash
#SBATCH --job-name=m3_ep2
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
cp $HOME/models/yolo_model_3/weights/best.pt $MODEL
# Directory for datasets
export DATASETS=$SCRATCH/datasets
mkdir -p $DATASETS
# Source to run detection on
cp -r $HOME/ep2_0f.tar $DATASETS/
cd $DATASETS
tar -xf ep2_0f.tar

# Change to $RUNDIR
cd $RUNDIR

# Run the file
echo "[$SHELL] Run script"
for F in $(find $DATASETS -mindepth 1 -maxdepth 1 -type d)
do
echo "Starting on ${F#${DATASETS}/}"
python detect.py --source $F/'**/*.jpg' --project $PROJECTDIR --name "ep2_${F#${DATASETS}/}" --img 1280 --weights $MODEL --device $CUDA_VISIBLE_DEVICES --save-txt --save-conf
done
echo "[$SHELL] Script finished"

# Move stat directory back to CWD
echo "[$SHELL] Remove dataset of images"
rm -rf $DATASETS
echo "[$SHELL] Zip and copy files back to cwd"
cd $PROJECTDIR
zip -r $SCRATCH/ep2_$SLURM_JOB_ID.zip ./ 
cp -r $SCRATCH/ep2_$SLURM_JOB_ID.zip $CWD/

echo "[$SHELL] #### Finished detection."
