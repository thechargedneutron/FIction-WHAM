#!/bin/bash

#SBATCH --job-name=extract_wham
#SBATCH --partition=enter
#SBATCH --nodes=1
#SBATCH --array=1-1438
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node! 1437
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 48:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/ex_%A_%a.out           # output file name
#SBATCH --error=logs/ex_%A_%a.err
#SBATCH --constraint='volta32gb'
#SBATCH --mem=60G
#SBATCH --open-mode=append

#SBATCH --mail-user=kumar.ashutosh.ee@gmail.com
#SBATCH --mail-type=begin,end,fail # mail once the job finishes




# Specify the file containing the list of takes
FILE="/path/to/Detic/all_takes_list.txt"

# Get the i'th line from the file based on the SLURM array task ID
TAKE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $FILE)

echo "Current take is $TAKE"

# Define the directory path for the current take
TAKE_DIR="/datasets01/egoexo4d/v2/takes/${TAKE}/frame_aligned_videos"

# List all mp4 files with "cam" in the name within the take directory
VIDEO_FILES=$(ls ${TAKE_DIR}/*{cam,gp}*.mp4 2>/dev/null)

# Check if any video files were found
if [ -z "$VIDEO_FILES" ]; then
  echo "No video files found for take ${TAKE}."
fi

# Loop over each video file and run the Python script
for VIDEO in $VIDEO_FILES; do
  echo "Processing video: $VIDEO"
  python demo.py --video "$VIDEO" --output_pth /path/to/temp_data/WHAM2/$TAKE --visualize --estimate_local_only --save_pkl
done