#!/bin/sh

cd ..

### DEM_cuboids
# MLP
./main.py \
        --is_loaded False \
        --config_path configs/DEM_cuboids/train_cuboids_mlp.json \
        --dataset_dir data/DEM_cuboids.csv \
        --data_is_aligned False \
        --rotate_columns True \
        --normalization_constant 1.0 \
        --additive_noise_stddev 1.12e-4 \
        --num_train_epochs 3000 \
        --batch_size 64 \
        --evaluate_every_n_epochs 20 \
        --lr_decay_after_epochs 300 \
        --lr_decay_factor 0.5 \
        --result_path results/DEM_cuboids/ \
        --execute_evaluation False \
        --execute_multi_target_tracking False \
        --no_show True
# RNN
./main.py \
        --is_loaded False \
        --config_path configs/DEM_cuboids/train_cuboids_rnn.json \
        --dataset_dir data/DEM_cuboids.csv \
        --data_is_aligned False \
        --rotate_columns True \
        --normalization_constant 1.0 \
        --additive_noise_stddev 1.12e-4 \
        --num_train_epochs 1000 \
        --batch_size 64 \
        --evaluate_every_n_epochs 20 \
        --lr_decay_after_epochs 150 \
        --lr_decay_factor 0.1 \
        --result_path results/DEM_cuboids/ \
        --execute_evaluation False \
        --execute_multi_target_tracking False \
        --no_show True

### DEM_cylinder
# MLP
./main.py \
        --is_loaded False \
        --config_path configs/DEM_cylinder/train_cylinder_mlp.json \
        --dataset_dir data/DEM_cylinder.csv \
        --data_is_aligned False \
        --rotate_columns True \
        --normalization_constant 1.0 \
        --additive_noise_stddev 1.12e-4 \
        --num_train_epochs 3000 \
        --batch_size 64 \
        --evaluate_every_n_epochs 20 \
        --lr_decay_after_epochs 300 \
        --lr_decay_factor 0.5 \
        --result_path results/DEM_cylinder/ \
        --execute_evaluation False \
        --execute_multi_target_tracking False \
        --no_show True
# RNN
./main.py \
        --is_loaded False \
        --config_path configs/DEM_cylinder/train_cylinder_rnn.json \
        --dataset_dir data/DEM_cylinder.csv \
        --data_is_aligned False \
        --rotate_columns True \
        --normalization_constant 1.0 \
        --additive_noise_stddev 1.12e-4 \
        --num_train_epochs 1000 \
        --batch_size 64 \
        --evaluate_every_n_epochs 20 \
        --lr_decay_after_epochs 150 \
        --lr_decay_factor 0.1 \
        --result_path results/DEM_cylinder/ \
        --execute_evaluation False \
        --execute_multi_target_tracking False \
        --no_show True

### DEM_spheres
# MLP
./main.py \
        --is_loaded False \
        --config_path configs/DEM_spheres/train_spheres_mlp.json \
        --dataset_dir data/DEM_spheres.csv \
        --data_is_aligned False \
        --rotate_columns True \
        --normalization_constant 1.0 \
        --additive_noise_stddev 1.12e-4 \
        --num_train_epochs 3000 \
        --batch_size 64 \
        --evaluate_every_n_epochs 20 \
        --lr_decay_after_epochs 300 \
        --lr_decay_factor 0.5 \
        --result_path results/DEM_spheres/ \
        --execute_evaluation False \
        --execute_multi_target_tracking False \
        --no_show True
# RNN
./main.py \
        --is_loaded False \
        --config_path configs/DEM_spheres/train_spheres_rnn.json \
        --dataset_dir data/DEM_spheres.csv \
        --data_is_aligned False \
        --rotate_columns True \
        --normalization_constant 1.0 \
        --additive_noise_stddev 1.12e-4 \
        --num_train_epochs 1000 \
        --batch_size 64 \
        --evaluate_every_n_epochs 20 \
        --lr_decay_after_epochs 150 \
        --lr_decay_factor 0.1 \
        --result_path results/DEM_spheres/ \
        --execute_evaluation False \
        --execute_multi_target_tracking False \
        --no_show True
