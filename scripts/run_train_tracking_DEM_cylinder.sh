#!/bin/sh

cd ..

./main.py \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/DEM_cylinder/train_cylinder_kf_cv.json \
  --dataset_dir 'data/DEM_Data/csv_converted_FOV/Cylinders_115_200Hz.csv' \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 4e-4 \
  --num_train_epochs 1 \
  --batch_size 128 \
  --evaluate_every_n_epochs 20 \
  --result_path results/DEM_cylinder/MTT_kf_cv/ \
  --execute_evaluation True \
  --execute_multi_target_tracking True \
  --no_show True \
  --logfile logs/terminal_logs/run_train_tracking_DEM_cylinder.log

./main.py \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/DEM_cylinder/train_cylinder_kf_ca.json \
  --dataset_dir 'data/DEM_Data/csv_converted_FOV/Cylinders_115_200Hz.csv' \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 4e-4 \
  --num_train_epochs 1 \
  --batch_size 128 \
  --evaluate_every_n_epochs 20 \
  --result_path results/DEM_cylinder/MTT_kf_ca/ \
  --execute_evaluation True \
  --execute_multi_target_tracking True \
  --no_show True \
  --logfile logs/terminal_logs/run_train_tracking_DEM_cylinder.log

./main.py \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/DEM_cylinder/train_cylinder_mlp.json \
  --dataset_dir 'data/DEM_Data/csv_converted_FOV/Cylinders_115_200Hz.csv' \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 4e-4 \
  --num_train_epochs 3000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 20 \
  --result_path results/DEM_cylinder/MTT_mlp/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_tracking_DEM_cylinder.log

./main.py \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/DEM_cylinder/train_cylinder_rnn.json \
  --dataset_dir 'data/DEM_Data/csv_converted_FOV/Cylinders_115_200Hz.csv' \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 4e-4 \
  --num_train_epochs 1000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 20 \
  --result_path results/DEM_cylinder/MTT_lstm/ \
  --execute_evaluation True \
  --execute_multi_target_tracking True \
  --no_show True \
  --logfile logs/terminal_logs/run_train_tracking_DEM_cylinder.log

./main.py \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/DEM_cylinder/train_cylinder_me_pos_id.json \
  --dataset_dir 'data/DEM_Data/csv_converted_FOV/Cylinders_115_200Hz.csv' \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 4e-4 \
  --num_train_epochs 3000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 20 \
  --result_path results/DEM_cylinder/MTT_me_pos_id/ \
  --execute_evaluation True \
  --execute_multi_target_tracking True \
  --no_show True \
  --logfile logs/terminal_logs/run_train_tracking_DEM_cylinder.log
