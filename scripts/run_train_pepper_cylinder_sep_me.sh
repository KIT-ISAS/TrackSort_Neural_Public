#!/bin/sh

cd ..

# ME
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/pepper_and_cylinder/train_pepper_cylinder_sep_gating_me.json \
  --dataset_dir 'data/pepper_and_cylinder/*_trackHistory_NothingDeleted.csv' \
  --result_path results/pepper_cylinder/separation_prediction_gating_me/ \
  --time_normalization_constant 15.71 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 3000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True
