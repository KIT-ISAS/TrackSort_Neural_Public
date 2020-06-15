#!/bin/sh

cd ..


./main.py \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/DEM_cylinder/train_cylinder_mlp.json \
  --dataset_dir 'data/DEM_Data/csv_converted_FOV/Cylinders_115_200Hz.csv' \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 1.12e-4 \
  --num_train_epochs 3000 \
  --batch_size 64 \
  --evaluate_every_n_epochs 20 \
  --result_path results/DEM_cylinder/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True

./main.py \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/DEM_cylinder/train_cylinder_rnn.json \
  --dataset_dir 'data/DEM_Data/csv_converted_FOV/Cylinders_115_200Hz.csv' \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 1.12e-4 \
  --num_train_epochs 3000 \
  --batch_size 64 \
  --evaluate_every_n_epochs 20 \
  --result_path results/DEM_cylinder/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True

./main.py \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/DEM_cylinder/test_cylinder_me_pos_id.json \
  --dataset_dir 'data/DEM_Data/csv_converted_FOV/Cylinders_115_200Hz.csv' \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 1.12e-4 \
  --num_train_epochs 3000 \
  --batch_size 64 \
  --evaluate_every_n_epochs 20 \
  --result_path results/DEM_cylinder/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True
