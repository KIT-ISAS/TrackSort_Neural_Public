#!/bin/sh
cd ..
## Train separation prediction on cylinder data

# MLP
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network False \
  --config_path configs/cylinder/train_cylinder_mlp_sep.json \
  --dataset_dir 'data/Zylinder/trackSortResultZylinder/*_trackHistory_NothingDeleted.csv' \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --num_train_epochs 1000 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.10 \
  --evaluate_every_n_epochs 100 \
  --time_normalization_constant 15.71 \
  --result_path results/cylinder/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_separation_cylinder.log

# RNN
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network False \
  --config_path configs/cylinder/train_cylinder_rnn_hyb.json \
  --dataset_dir 'data/Zylinder/trackSortResultZylinder/*_trackHistory_NothingDeleted.csv' \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --num_train_epochs 1000 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.10 \
  --evaluate_every_n_epochs 20 \
  --time_normalization_constant 15.71 \
  --result_path results/cylinder/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_separation_cylinder.log

# KF
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network False \
  --config_path configs/cylinder/train_cylinder_sep_kf.json \
  --dataset_dir 'data/Zylinder/trackSortResultZylinder/*_trackHistory_NothingDeleted.csv' \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --num_train_epochs 1 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.10 \
  --evaluate_every_n_epochs 100 \
  --time_normalization_constant 15.71 \
  --result_path results/cylinder/separation_prediction_default/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_separation_cylinder.log

# Mixture of Experts
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/cylinder/train_cylinder_sep_gating_me.json \
  --dataset_dir 'data/Zylinder/trackSortResultZylinder/*_trackHistory_NothingDeleted.csv' \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --num_train_epochs 1000 \
  --evaluation_ratio 0.15 \
  --test_ratio 0.10 \
  --evaluate_every_n_epochs 100 \
  --time_normalization_constant 15.71 \
  --result_path results/cylinder/separation_prediction_gating_me/ \
  --execute_evaluation False \
  --execute_multi_target_tracking False \
  --no_show True \
  --logfile logs/terminal_logs/run_train_separation_cylinder.log
