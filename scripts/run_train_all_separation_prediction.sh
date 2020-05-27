#!/bin/sh
# Train all models (all KF, RNN and MLP) on all Datasets (Real: Pepper, Spheres, Wheat; DEM: Cylinder, Cuboids)
cd ..

## Data: Pepper
# MLP
./main.py \
    --separation_prediction True \
    --tracking False \
    --virtual_belt_edge_x_position 800 \
    --virtual_nozzle_array_x_position 1550 \
    --is_loaded False \
    --config_path configs/pepper/train_pepper_mlp_sep.json \
    --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
    --separation_mlp_input_dim 7 \
    --num_train_epochs 2000 \
    --batch_size 64 \
    --evaluation_ratio 0.15 \
    --test_ratio 0.15 \
    --evaluate_every_n_epochs 20 \
    --lr_decay_after_epochs 300 \
    --lr_decay_factor 0.5 \
    --time_normalization_constant 22 \
    --result_path results/pepper_all/separation_prediction_default/ \
    --execute_evaluation False \
    --execute_multi_target_tracking False \
    --no_show True \
    --logfile logs/terminal_logs/run_train_all_separation_prediction.log

# RNN
./main.py \
    --separation_prediction True \
    --tracking False \
    --virtual_belt_edge_x_position 800 \
    --virtual_nozzle_array_x_position 1550 \
    --is_loaded False \
    --config_path configs/pepper/train_pepper_rnn_hyb.json \
    --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
    --num_train_epochs 1000 \
    --batch_size 64 \
    --evaluation_ratio 0.15 \
    --test_ratio 0.15 \
    --evaluate_every_n_epochs 20 \
    --lr_decay_after_epochs 150 \
    --lr_decay_factor 0.1 \
    --time_normalization_constant 22 \
    --result_path results/pepper_all/separation_prediction_default/ \
    --execute_evaluation False \
    --execute_multi_target_tracking False \
    --no_show True \
    --logfile logs/terminal_logs/run_train_all_separation_prediction.log

  # KF
  ./main.py \
    --separation_prediction True \
    --tracking False \
    --virtual_belt_edge_x_position 800 \
    --virtual_nozzle_array_x_position 1550 \
    --is_loaded False \
    --config_path configs/pepper/train_pepper_sep_kf.json \
    --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
    --num_train_epochs 1 \
    --batch_size 64 \
    --evaluation_ratio 0.15 \
    --test_ratio 0.15 \
    --time_normalization_constant 22 \
    --result_path results/pepper_all/separation_prediction_default/ \
    --execute_evaluation False \
    --execute_multi_target_tracking False \
    --no_show True \
    --logfile logs/terminal_logs/run_train_all_separation_prediction.log

## Data: Spheres
# MLP
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --config_path configs/spheres/train_spheres_mlp_sep.json \
  --dataset_dir 'data/Kugeln/trackSortResultKugeln/*_trackHistory_NothingDeleted.csv' \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 2000 \
  --batch_size 64 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.15 \
  --evaluate_every_n_epochs 20 \
  --lr_decay_after_epochs 300 \
  --lr_decay_factor 0.5 \
  --time_normalization_constant 22 \
  --result_path results/spheres_all/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_all_separation_prediction.log

# RNN
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --config_path configs/spheres/train_spheres_rnn_hyb.json \
  --dataset_dir 'data/Kugeln/trackSortResultKugeln/*_trackHistory_NothingDeleted.csv' \
  --num_train_epochs 1000 \
  --batch_size 64 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.15 \
  --evaluate_every_n_epochs 20 \
  --lr_decay_after_epochs 150 \
  --lr_decay_factor 0.1 \
  --time_normalization_constant 22 \
  --result_path results/spheres_all/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_all_separation_prediction.log

# KF
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --config_path configs/spheres/train_spheres_sep_kf.json \
  --dataset_dir 'data/Kugeln/trackSortResultKugeln/*_trackHistory_NothingDeleted.csv' \
  --num_train_epochs 1 \
  --batch_size 64 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.15 \
  --time_normalization_constant 22 \
  --result_path results/spheres_all/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_all_separation_prediction.log

## Data: Wheat
# MLP
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --config_path configs/wheat/train_wheat_mlp_sep.json \
  --dataset_dir 'data/Weizen/trackSortResultWeizen/*_trackHistory_NothingDeleted.csv' \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 2000 \
  --batch_size 64 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.15 \
  --evaluate_every_n_epochs 20 \
  --lr_decay_after_epochs 300 \
  --lr_decay_factor 0.5 \
  --time_normalization_constant 22 \
  --result_path results/wheat_all/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_all_separation_prediction.log

# RNN
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --config_path configs/wheat/train_wheat_rnn_hyb.json \
  --dataset_dir 'data/Weizen/trackSortResultWeizen/*_trackHistory_NothingDeleted.csv' \
  --num_train_epochs 1000 \
  --batch_size 64 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.15 \
  --evaluate_every_n_epochs 20 \
  --lr_decay_after_epochs 150 \
  --lr_decay_factor 0.1 \
  --time_normalization_constant 22 \
  --result_path results/wheat_all/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_all_separation_prediction.log

# KF
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --config_path configs/wheat/train_wheat_sep_kf.json \
  --dataset_dir 'data/Weizen/trackSortResultWeizen/*_trackHistory_NothingDeleted.csv' \
  --num_train_epochs 1 \
  --batch_size 64 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.15 \
  --time_normalization_constant 22 \
  --result_path results/wheat_all/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_all_separation_prediction.log

## Data: DEM cuboids
# MLP
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 0.55 \
  --virtual_nozzle_array_x_position 0.7 \
  --is_loaded False \
  --config_path configs/DEM_cuboids/train_cuboids_mlp_sep.json \
  --dataset_dir 'data/DEM_Data/csv_converted/Cuboids_115_200Hz.csv' \
  --dataset_type CsvDataset \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 1.12e-4 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 2000 \
  --batch_size 64 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.15 \
  --evaluate_every_n_epochs 20 \
  --lr_decay_after_epochs 300 \
  --lr_decay_factor 0.5 \
  --time_normalization_constant 22 \
  --result_path results/DEM_cuboids/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_all_separation_prediction.log

# RNN
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --config_path configs/DEM_cuboids/train_cuboids_rnn_hyb.json \
  --dataset_dir 'data/DEM_Data/csv_converted/Cuboids_115_200Hz.csv' \
  --dataset_type CsvDataset \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 1.12e-4 \
  --num_train_epochs 1000 \
  --batch_size 64 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.15 \
  --evaluate_every_n_epochs 20 \
  --lr_decay_after_epochs 150 \
  --lr_decay_factor 0.1 \
  --time_normalization_constant 22 \
  --result_path results/DEM_cuboids/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_all_separation_prediction.log

# KF
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --config_path configs/DEM_cuboids/train_cuboids_sep_kf.json \
  --dataset_dir 'data/DEM_Data/csv_converted/Cuboids_115_200Hz.csv' \
  --dataset_type CsvDataset \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 1.12e-4 \
  --num_train_epochs 1 \
  --batch_size 64 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.15 \
  --time_normalization_constant 22 \
  --result_path results/DEM_cuboids/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_all_separation_prediction.log

## Data: DEM cylinder
# MLP
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 0.55 \
  --virtual_nozzle_array_x_position 0.7 \
  --is_loaded False \
  --config_path configs/DEM_cylinder/train_cylinder_mlp_sep.json \
  --dataset_dir 'data/DEM_Data/csv_converted/Cylinders_115_200Hz.csv' \
  --dataset_type CsvDataset \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 1.12e-4 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 2000 \
  --batch_size 64 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.15 \
  --evaluate_every_n_epochs 20 \
  --lr_decay_after_epochs 300 \
  --lr_decay_factor 0.5 \
  --time_normalization_constant 22 \
  --result_path results/DEM_cylinder/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_all_separation_prediction.log

# RNN
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --config_path configs/DEM_cylinder/train_cylinder_rnn_hyb.json \
  --dataset_dir 'data/DEM_Data/csv_converted/Cylinders_115_200Hz.csv' \
  --dataset_type CsvDataset \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 1.12e-4 \
  --num_train_epochs 1000 \
  --batch_size 64 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.15 \
  --evaluate_every_n_epochs 20 \
  --lr_decay_after_epochs 150 \
  --lr_decay_factor 0.1 \
  --time_normalization_constant 22 \
  --result_path results/DEM_cylinder/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_all_separation_prediction.log

# KF
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --config_path configs/DEM_cylinder/train_cylinder_sep_kf.json \
  --dataset_dir 'data/DEM_Data/csv_converted/Cylinders_115_200Hz.csv' \
  --dataset_type CsvDataset \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 1.12e-4 \
  --num_train_epochs 1 \
  --batch_size 64 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.15 \
  --time_normalization_constant 22 \
  --result_path results/DEM_cylinder/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_all_separation_prediction.log
