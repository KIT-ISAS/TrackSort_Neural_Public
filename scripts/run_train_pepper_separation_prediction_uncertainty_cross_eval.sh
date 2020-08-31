#!/bin/sh
# Train all models (all KF, RNN and MLP) on the pepper dataset with uncertainty prediction and cross evaluation
cd ..

## Data: Pepper
# MLP
#./main.py \
#    --separation_prediction True \
#    --tracking False \
#    --uncertainty_prediction True \
#    --virtual_belt_edge_x_position 800 \
#    --virtual_nozzle_array_x_position 1550 \
#    --is_loaded False \
#    --is_loaded_gating_network False \
#    --n_folded_cross_evaluation 5 \
#    --config_path configs/pepper/train_pepper_mlp_sep_uncertainty.json \
#    --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
#    --separation_mlp_input_dim 7 \
#    --num_train_epochs 4000 \
#    --batch_size 128 \
#    --evaluate_every_n_epochs 500 \
#    --time_normalization_constant 15.71 \
#    --result_path results/pepper_all/separation_prediction_mlp_uncertainty/ \
#    --execute_evaluation True \
#    --execute_multi_target_tracking False \
#    --no_show True \
#    --logfile logs/terminal_logs/run_train_pepper_separation_prediction_uncertainty_cross_eval.log

# RNN
./main.py \
    --separation_prediction True \
    --tracking False \
    --uncertainty_prediction True \
    --virtual_belt_edge_x_position 800 \
    --virtual_nozzle_array_x_position 1550 \
    --is_loaded False \
    --is_loaded_gating_network False \
    --n_folded_cross_evaluation 5 \
    --config_path configs/pepper/train_pepper_rnn_hyb_uncertainty.json \
    --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
    --num_train_epochs 3000 \
    --batch_size 128 \
    --evaluate_every_n_epochs 100 \
    --time_normalization_constant 15.71 \
    --result_path results/pepper_all/separation_prediction_rnn_uncertainty/ \
    --execute_evaluation True \
    --execute_multi_target_tracking False \
    --no_show True \
    --logfile logs/terminal_logs/run_train_pepper_separation_prediction_uncertainty_cross_eval.log
