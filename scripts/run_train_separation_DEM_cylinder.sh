#!/bin/sh
cd ..
## Train separation prediction on DEM cylinder data

# MLP
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 0.7 \
  --virtual_nozzle_array_x_position 0.741 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/DEM_cylinder/train_cylinder_mlp_sep.json \
  --dataset_dir 'data/DEM_Data/csv_converted_FOV/Cylinders_115_200Hz.csv' \
  --dataset_type CsvDataset \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 1.12e-4 \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --num_train_epochs 1000 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.10 \
  --evaluate_every_n_epochs 100 \
  --time_normalization_constant 15.71 \
  --result_path results/DEM_cuboids/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_separation_DEM_cylinders.log

# RNN
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 0.7 \
  --virtual_nozzle_array_x_position 0.741 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/DEM_cylinder/train_cylinder_rnn_hyb.json \
  --dataset_dir 'data/DEM_Data/csv_converted_FOV/Cylinders_115_200Hz.csv' \
  --dataset_type CsvDataset \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 1.12e-4 \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --num_train_epochs 1000 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.10 \
  --evaluate_every_n_epochs 100 \
  --time_normalization_constant 15.71 \
  --result_path results/DEM_cuboids/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_separation_DEM_cylinders.log

# KF
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 0.7 \
  --virtual_nozzle_array_x_position 0.741 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/DEM_cylinder/train_cylinder_sep_kf.json \
  --dataset_dir 'data/DEM_Data/csv_converted_FOV/Cylinders_115_200Hz.csv' \
  --dataset_type CsvDataset \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 1.12e-4 \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --num_train_epochs 1 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.10 \
  --evaluate_every_n_epochs 100 \
  --time_normalization_constant 15.71 \
  --result_path results/DEM_cuboids/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_separation_DEM_cylinders.log

# Mixture of Experts
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 0.7 \
  --virtual_nozzle_array_x_position 0.741 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/DEM_cylinder/train_cylinder_sep_gating_me.json \
  --dataset_dir 'data/DEM_Data/csv_converted_FOV/Cylinders_115_200Hz.csv' \
  --dataset_type CsvDataset \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 1.12e-4 \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --num_train_epochs 1000 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.10 \
  --evaluate_every_n_epochs 100 \
  --time_normalization_constant 15.71 \
  --result_path results/DEM_cuboids/separation_prediction_gating_me/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_separation_DEM_cylinders.log
