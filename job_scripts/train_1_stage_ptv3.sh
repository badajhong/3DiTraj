#!/bin/bash

set -x
set -e

# Go to your project directory
cd $HOME/Desktop/robot-3dlotus

# Add current directory to PYTHONPATH
export PYTHONPATH=$HOME/Desktop/robot-3dlotus:$PYTHONPATH

# Activate your conda environment
. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate gembench

# Set defaults (not needed but kept for compatibility)
export MASTER_PORT=12345
export WORLD_SIZE=1
export MASTER_ADDR=localhost

# Increase file handle limit
ulimit -n 2048

# Hyperparams
output_dir=data/experiments/gembench/3dlotus/v4
rot_type=quat
npoints=4096
pos_bin_size=15

# Run training script directly (no srun!)
python genrobo3d/train/train_1_stage_ptv3.py \
    --exp-config genrobo3d/configs/rlbench/1_stage_ptv3.yaml \
    output_dir ${output_dir} \