#!/bin/bash
#SBATCH --job-name=r_m4_all
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-user="b.a.companjen@library.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=5G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gpus=1

# load modules (assuming you start from the default environment)
# we explicitely call the modules to improve reproducability
# in case the default settings change
module load torchvision/0.8.2-fosscuda-2020b-PyTorch-1.7.1
module load rclone
module load FFmpeg

source $HOME/venvs/yolo_env/bin/activate

echo "[$SHELL] #### Starting GPU YOLOv5 detection"
echo "[$SHELL] This job has the ID $SLURM_JOB_ID and task ID $SLURM_ARRAY_TASK_ID"
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
export SD_EPI=$(sed -n ${SLURM_ARRAY_TASK_ID}p $CWD/episodes.txt)
# export SD_EPI=$(sed -n ${BLA}p $CWD/episodes.txt)
echo "[$SHELL] File to download from SURFdrive: "$SD_EPI

echo "[$SHELL] Copying the YOLO model to scratch..."
export MODEL=$SCRATCH/yolo_1280.pt
cp $HOME/models/yolo_model_4/weights/best.pt $MODEL
# Directory for datasets
export DATASETS=$SCRATCH/datasets
mkdir -p $DATASETS
export VIDEO=$DATASETS/ep${SLURM_ARRAY_TASK_ID}.mp4
export VIDEO_FF=$DATASETS/ep${SLURM_ARRAY_TASK_ID}_0f.mp4
# Source to run detection on
echo "[$SHELL] Downloading video file..."

rclone copy "SD:${SD_EPI}" $DATASETS/
ls -lh $DATASETS
export OVIDEO=$(basename "${SD_EPI}")
mv "$DATASETS/$OVIDEO" $VIDEO
echo "[$SHELL] Done downloading video file!"
ls -lh $VIDEO

echo "[$SHELL] Converting video file to ${VIDEO_FF} with fewer frames..."
time ffmpeg -hide_banner -loglevel error -i ${VIDEO} -vsync passthrough -copyts -lavfi select='not(mod(n\,10))' -an -qscale:v 1 -qmin 1 -qmax 1 ${VIDEO_FF}
# Change to $RUNDIR
cd $RUNDIR

# Run the file
echo "[$SHELL] Run script"

python detect.py --source $VIDEO_FF --project $PROJECTDIR --name "ep${SLURM_ARRAY_TASK_ID}" --img 1280 --weights $MODEL --device $CUDA_VISIBLE_DEVICES --save-txt --save-conf

echo "[$SHELL] Script finished"

echo "[$SHELL] Zip and copy files back to SURFdrive"
cd $PROJECTDIR/ep${SLURM_ARRAY_TASK_ID}
zip -r ep${SLURM_ARRAY_TASK_ID}_$SLURM_JOB_ID.zip ./labels
# Now we have the output video and a zip file of labels, we can remove the
# folder of labels
rm -rf ./labels
# Copy the directory contents to SURFdrive
rclone copy . SD:ProjectM/arrayjob --timeout 50m --use-cookies

echo "[$SHELL] #### Finished task!"
