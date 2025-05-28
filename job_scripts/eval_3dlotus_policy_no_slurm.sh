#!/bin/bash

set -x
set -e

# Go to your project directory
cd $HOME/Desktop/robot-3dlotus

# Activate your conda environment
. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate gembench

expr_dir=data/experiments/gembench/3dlotus/v2
ckpt_step=40000
exp_config=data/experiments/gembench/3dlotus/v2/logs/training_config.yaml   # <-- SET THIS TO YOUR CONFIG FILE

# # Run validation
# python genrobo3d/evaluation/eval_simple_policy.py \
#     --exp_config ${exp_config} \
#     --checkpoint ${expr_dir}/ckpts/model_step_${ckpt_step}.pt \
#     --num_demos 20 \
#     --seed 100 \
#     --microstep_data_dir data/gembench/val_dataset/microsteps/seed100 \
#     --taskvar push_button+0
#     --headless

# Run tests
for seed in {200..600..100}
do
  for split in train test_l2 test_l3 test_l4
  do
    for taskvar in $(jq -r '.[]' assets/taskvars_${split}.json)
    do
      python genrobo3d/evaluation/eval_simple_policy.py \
          --exp_config ${exp_config} \
          --checkpoint ${expr_dir}/ckpts/model_step_${ckpt_step}.pt \
          --num_demos 20 \
          --seed ${seed} \
          --microstep_data_dir data/gembench/test_dataset/microsteps/seed${seed} \
          --taskvar ${taskvar}
          # --headless
    done
  done
done