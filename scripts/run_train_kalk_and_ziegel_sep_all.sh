#!/bin/sh

cd ..

## kalk
# MLP
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/kalk/train_kalk_mlp_sep_uncertainty.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --result_path results/kalk_all/separation_prediction_mlp_uncertainty/ \
  --separation_mlp_input_dim 7 \
  --time_normalization_constant 15.71 \
  --num_train_epochs 4000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# LSTM
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/kalk/train_kalk_rnn_hyb_uncertainty.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --result_path results/kalk_all/separation_prediction_lstm_uncertainty/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# KF
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/kalk/train_kalk_kf_sep.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --result_path results/kalk_all/separation_prediction_kf/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# ME
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/kalk/train_kalk_sep_gating_me.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --result_path results/kalk_all/separation_prediction_gating_me/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 3000 \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# SE
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/kalk/train_kalk_sep_gating_se.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --result_path results/kalk_all/separation_prediction_gating_se/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1 \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# COV
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/kalk/train_kalk_sep_gating_cov.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --result_path results/kalk_all/separation_prediction_gating_cov/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1 \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

## ziegel
# MLP
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/ziegel/train_ziegel_mlp_sep_uncertainty.json \
  --dataset_dir 'data/ziegel/trackSortResultziegel/*_trackHistory_NothingDeleted.csv' \
  --result_path results/ziegel_all/separation_prediction_mlp_uncertainty/ \
  --separation_mlp_input_dim 7 \
  --time_normalization_constant 15.71 \
  --num_train_epochs 4000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# LSTM
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/ziegel/train_ziegel_rnn_hyb_uncertainty.json \
  --dataset_dir 'data/ziegel/trackSortResultziegel/*_trackHistory_NothingDeleted.csv' \
  --result_path results/ziegel_all/separation_prediction_lstm_uncertainty/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# KF
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/ziegel/train_ziegel_kf_sep.json \
  --dataset_dir 'data/ziegel/trackSortResultziegel/*_trackHistory_NothingDeleted.csv' \
  --result_path results/ziegel_all/separation_prediction_kf/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# ME
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/ziegel/train_ziegel_sep_gating_me.json \
  --dataset_dir 'data/ziegel/trackSortResultziegel/*_trackHistory_NothingDeleted.csv' \
  --result_path results/ziegel_all/separation_prediction_gating_me/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 3000 \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# SE
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/ziegel/train_ziegel_sep_gating_se.json \
  --dataset_dir 'data/ziegel/trackSortResultziegel/*_trackHistory_NothingDeleted.csv' \
  --result_path results/ziegel_all/separation_prediction_gating_se/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1 \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# COV
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/ziegel/train_ziegel_sep_gating_cov.json \
  --dataset_dir 'data/ziegel/trackSortResultziegel/*_trackHistory_NothingDeleted.csv' \
  --result_path results/ziegel_all/separation_prediction_gating_cov/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1 \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

##### ME ######
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/kalk_and_ziegel/train_kalk_ziegel_sep_gating_me.json \
  --dataset_dir 'data/kalk_and_ziegel/*_trackHistory_NothingDeleted.csv' \
  --result_path results/kalk_ziegel/separation_prediction_gating_me/ \
  --time_normalization_constant 15.71 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 3000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True
