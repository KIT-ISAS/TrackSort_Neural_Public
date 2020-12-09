#!/bin/sh

cd ..

## PEPPER
# MLP
./main.py \
  --separation_prediction True \
  --tracking False \
  --uncertainty_prediction True \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/pepper/train_pepper_mlp_sep_uncertainty.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --result_path results/pepper_all/separation_prediction_mlp_uncertainty/ \
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
  --uncertainty_prediction True \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/pepper/train_pepper_rnn_hyb_uncertainty.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --result_path results/pepper_all/separation_prediction_lstm_uncertainty/ \
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
  --config_path configs/pepper/train_pepper_kf_sep.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --result_path results/pepper_all/separation_prediction_kf/ \
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
  --config_path configs/pepper/train_pepper_sep_gating_me.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --result_path results/pepper_all/separation_prediction_gating_me/ \
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
  --config_path configs/pepper/train_pepper_sep_gating_se.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --result_path results/pepper_all/separation_prediction_gating_se/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1 \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# SMAPE
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/pepper/train_pepper_sep_gating_smape.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --result_path results/pepper_all/separation_prediction_gating_smape/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1 \
  --separation_mlp_input_dim 7 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

## CYLINDER
# MLP
./main.py \
  --separation_prediction True \
  --tracking False \
  --uncertainty_prediction True \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/cylinder/train_cylinder_mlp_sep_uncertainty.json \
  --dataset_dir 'data/Zylinder/trackSortResultZylinder/*_trackHistory_NothingDeleted.csv' \
  --result_path results/cylinder_all/separation_prediction_mlp_uncertainty/ \
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
  --uncertainty_prediction True \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/cylinder/train_cylinder_rnn_hyb_uncertainty.json \
  --dataset_dir 'data/Zylinder/trackSortResultZylinder/*_trackHistory_NothingDeleted.csv' \
  --result_path results/cylinder_all/separation_prediction_lstm_uncertainty/ \
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
  --config_path configs/cylinder/train_cylinder_kf_sep.json \
  --dataset_dir 'data/Zylinder/trackSortResultZylinder/*_trackHistory_NothingDeleted.csv' \
  --result_path results/cylinder_all/separation_prediction_kf/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

## SPHERES
# MLP
./main.py \
  --separation_prediction True \
  --tracking False \
  --uncertainty_prediction True \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/spheres/train_spheres_mlp_sep_uncertainty.json \
  --dataset_dir 'data/Kugeln/trackSortResultKugeln/*_trackHistory_NothingDeleted.csv' \
  --result_path results/spheres_all/separation_prediction_mlp_uncertainty/ \
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
  --uncertainty_prediction True \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/spheres/train_spheres_rnn_hyb_uncertainty.json \
  --dataset_dir 'data/Kugeln/trackSortResultKugeln/*_trackHistory_NothingDeleted.csv' \
  --result_path results/spheres_all/separation_prediction_lstm_uncertainty/ \
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
  --config_path configs/spheres/train_spheres_kf_sep.json \
  --dataset_dir 'data/Kugeln/trackSortResultKugeln/*_trackHistory_NothingDeleted.csv' \
  --result_path results/spheres_all/separation_prediction_kf/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

## WHEAT
# MLP
./main.py \
  --separation_prediction True \
  --tracking False \
  --uncertainty_prediction True \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/wheat/train_wheat_mlp_sep_uncertainty.json \
  --dataset_dir 'data/Weizen/trackSortResultWeizen/*_trackHistory_NothingDeleted.csv' \
  --result_path results/wheat_all/separation_prediction_mlp_uncertainty/ \
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
  --uncertainty_prediction True \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded False \
  --is_loaded_gating_network True \
  --config_path configs/wheat/train_wheat_rnn_hyb_uncertainty.json \
  --dataset_dir 'data/Weizen/trackSortResultWeizen/*_trackHistory_NothingDeleted.csv' \
  --result_path results/wheat_all/separation_prediction_lstm_uncertainty/ \
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
  --config_path configs/wheat/train_wheat_kf_sep.json \
  --dataset_dir 'data/Weizen/trackSortResultWeizen/*_trackHistory_NothingDeleted.csv' \
  --result_path results/wheat_all/separation_prediction_kf/ \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1 \
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
  --config_path configs/cylinder_pepper_spheres_wheat/train_cylinder_pepper_spheres_wheat_sep_gating_me.json \
  --dataset_dir 'data/cylinder_pepper_spheres_wheat/*_trackHistory_NothingDeleted.csv' \
  --result_path results/cylinder_pepper_spheres_wheat/separation_prediction_gating_me/ \
  --time_normalization_constant 15.71 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 3000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True
