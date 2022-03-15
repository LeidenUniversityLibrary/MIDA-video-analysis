#!/bin/bash
#SBATCH --job-name=scd_all
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-user="b.a.companjen@library.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gpus=1

# load modules (assuming you start from the default environment)
# we explicitely call the modules to improve reproducability
# in case the default settings change
module load rclone
module load FFmpeg

echo "[$SHELL] #### Starting scene detection"
echo "[$SHELL] This job has the ID $SLURM_ARRAY_JOB_ID and task ID $SLURM_ARRAY_TASK_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] CWD: "$CWD

# Which GPU has been assigned
echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES

# Create a directory of local scratch on the node
echo "[$SHELL] Node scratch: "$SCRATCH

# Find the episode's video to download
export SD_EPI=$(sed -n ${SLURM_ARRAY_TASK_ID}p $CWD/episodes.txt)
echo "[$SHELL] File to download from SURFdrive: "$SD_EPI

# Directory for datasets
export DATASETS=$SCRATCH/datasets
mkdir -p $DATASETS
export EPI=ep${SLURM_ARRAY_TASK_ID}
export VIDEO=$DATASETS/${EPI}.mp4
export FRAMES_FOLDER=$DATASETS/frames

# Source to run detection on
echo "[$SHELL] Downloading video file..."

rclone copy "SD:${SD_EPI}" $DATASETS/
ls -lh $DATASETS
export OVIDEO=$(basename "${SD_EPI}")
mv "$DATASETS/$OVIDEO" $VIDEO
echo "[$SHELL] Done downloading video file!"
ls -lh $VIDEO

#### Scene analysis
echo "[$SHELL] Analysing scene changes in ${VIDEO}..."

SCD_DIR=$SCRATCH/"scene-detection/"
mkdir -p "${SCD_DIR}"

OUTPUT_TXT="${SCD_DIR}/${EPI}_ffprobe-flat.txt"
SCD_THRESHOLD=4.5

export FFREPORT="file=${SCD_DIR}/${EPI}_ffprobe-report.log"
CMD="ffprobe -v error -f lavfi -i movie=${VIDEO},scdet=threshold=${SCD_THRESHOLD},blackframe=amount=99:threshold=24,signalstats -show_entries frame=pkt_pts_time,width,height:frame_tags -print_format flat"
${CMD} | sed -E -f ${HOME}/cleanup.txt > ${OUTPUT_TXT}
echo "[$SHELL] Zip scene detection results"
cd $SCD_DIR
zip -r $SCRATCH/${EPI}_${SLURM_ARRAY_JOB_ID}_scd.zip ./
# Copy the zip file to SURFdrive
echo "[$SHELL] Copy scene detection results to SURFdrive"
rclone copy $SCRATCH/${EPI}_${SLURM_ARRAY_JOB_ID}_scd.zip SD:ProjectM/arrayjob_frames --timeout 20m --use-cookies
# Unset FF* report file path
export FFREPORT=""

#### Frame extraction
echo "[$SHELL] Extracting frames from ${VIDEO}..."
LENGTH=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ${VIDEO})

number_of_frames=$(ffprobe -v error -show_entries stream=nb_frames -select_streams v -of default=noprint_wrappers=1:nokey=1 ${VIDEO})
echo "Total number of frames: ${number_of_frames}"
REDFACTOR=10
# It looks like all videos have the same 25 fps framerate, so we don't need to
# use it in calculating the number of seconds we need to extract to each folder
# framerate=$(ffprobe -v error -show_entries stream=r_frame_rate -select_streams v -of default=noprint_wrappers=1:nokey=1 ${FOLDER}/${VIDEO})
folders=$(( (${number_of_frames}/${REDFACTOR}) / 1000))
echo "Folders to create: 0 through "${folders}
iterations=$((${LENGTH%.*} / 400))
echo "Iterations to run: 0 through "$iterations
if [[ ${folders} -ne ${iterations} ]]; then
    echo "Something is wrong in the calculations, exiting"
    exit 1
fi

for START in $(seq 0 $iterations); do
    ST=$((${START} * 400))
    mkdir -p ${FRAMES_FOLDER}/${START}
    echo "Starting extraction at $ST seconds"
    ffmpeg -hide_banner -v error -ss ${ST}.0 -t 400.0 -i ${VIDEO} -vsync passthrough -copyts -lavfi select="not(mod(n\,${REDFACTOR}))" -frame_pts true -qscale:v 1 -qmin 1 -qmax 1 ${FRAMES_FOLDER}/${START}/f-%06d.jpg
done

# Copy the extracted frames to SURFdrive
cd ${FRAMES_FOLDER}
echo "[$SHELL] Tar extracted frames"
tar -cf $SCRATCH/${EPI}_0f.tar ./
echo "[$SHELL] Copy extracted frames to SURFdrive"
rclone copy $SCRATCH/${EPI}_0f.tar SD:ProjectM/extracted_frames --timeout 50m --use-cookies

echo "[$SHELL] #### Finished task!"
