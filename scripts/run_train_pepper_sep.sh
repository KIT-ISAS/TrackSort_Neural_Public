#!/bin/sh

cd ..

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
        --evaluate_every_n_epochs 20 \
        --lr_decay_after_epochs 300 \
        --lr_decay_factor 0.5 \
        --result_path results/pepper_all/separation_prediction_default/ \
        --execute_evaluation False \
        --execute_multi_target_tracking False \
        --no_show True

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
        --evaluate_every_n_epochs 20 \
        --lr_decay_after_epochs 150 \
        --lr_decay_factor 0.1 \
        --result_path results/pepper_all/separation_prediction_default/ \
        --execute_evaluation False \
        --execute_multi_target_tracking False \
        --no_show True
