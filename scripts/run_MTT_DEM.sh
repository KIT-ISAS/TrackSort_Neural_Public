#!/bin/sh

cd ..

### DEM_spheres
# MLP
./main.py \
        --is_loaded True \
        --config_path configs/DEM_spheres/train_spheres_mlp.json \
        --dataset_dir data/DEM_spheres.csv \
        --data_is_aligned False \
        --rotate_columns True \
        --normalization_constant 1.0 \
        --additive_noise_stddev 1.12e-4 \
        --batch_size 64 \
        --result_path results/DEM_spheres/MTT_MLP/ \
        --execute_evaluation False \
        --execute_multi_target_tracking True \
        --no_show True
# RNN
./main.py \
        --is_loaded True \
        --config_path configs/DEM_spheres/train_spheres_rnn.json \
        --dataset_dir data/DEM_spheres.csv \
        --data_is_aligned False \
        --rotate_columns True \
        --normalization_constant 1.0 \
        --additive_noise_stddev 1.12e-4 \
        --batch_size 64 \
        --result_path results/DEM_spheres/MTT_RNN/ \
        --execute_evaluation False \
        --execute_multi_target_tracking True \
        --no_show True
# KF CA
./main.py \
        --is_loaded True \
        --config_path configs/DEM_spheres/train_spheres_kf_ca.json \
        --dataset_dir data/DEM_spheres.csv \
        --data_is_aligned False \
        --rotate_columns True \
        --normalization_constant 1.0 \
        --additive_noise_stddev 1.12e-4 \
        --batch_size 64 \
        --result_path results/DEM_spheres/MTT_KF_CA/ \
        --execute_evaluation False \
        --execute_multi_target_tracking True \
        --no_show True
# SE
./main.py \
        --is_loaded True \
        --config_path configs/DEM_spheres/test_spheres_se.json \
        --dataset_dir data/DEM_spheres.csv \
        --data_is_aligned False \
        --rotate_columns True \
        --normalization_constant 1.0 \
        --additive_noise_stddev 1.12e-4 \
        --batch_size 64 \
        --result_path results/DEM_spheres/MTT_SE/ \
        --execute_evaluation False \
        --execute_multi_target_tracking True \
        --no_show True
# ME
./main.py \
        --is_loaded True \
        --config_path configs/DEM_spheres/test_spheres_me_pos_id.json \
        --dataset_dir data/DEM_spheres.csv \
        --data_is_aligned False \
        --rotate_columns True \
        --normalization_constant 1.0 \
        --additive_noise_stddev 1.12e-4 \
        --batch_size 64 \
        --result_path results/DEM_spheres/MTT_ME_POS_ID/ \
        --execute_evaluation False \
        --execute_multi_target_tracking True \
        --no_show True
