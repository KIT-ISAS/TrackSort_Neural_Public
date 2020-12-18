#!/bin/sh

cd ..

##### ME ######
# cylinder_pepper_spheres
mv data/cylinder_pepper_spheres_wheat/weizen* data/tmp_not_in_all/
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/crosseval_ME/train_cylinder_pepper_spheres_sep_gating_me.json \
  --dataset_dir 'data/cylinder_pepper_spheres_wheat/*_trackHistory_NothingDeleted.csv' \
  --result_path results/crosseval_ME/separation_prediction_gating_me_cylinder_pepper_spheres/ \
  --time_normalization_constant 15.71 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 3000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# wheat test
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network True \
  --config_path configs/crosseval_ME/train_cylinder_pepper_spheres_sep_gating_me.json \
  --dataset_dir 'data/tmp_not_in_all/*_trackHistory_NothingDeleted.csv' \
  --result_path results/crosseval_ME/separation_prediction_gating_me_test_wheat/ \
  --time_normalization_constant 15.71 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 3000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True \
  --evaluation_ratio 0.05 \
  --test_ratio 0.9

# cylinder_pepper_wheat
mv data/tmp_not_in_all/* data/cylinder_pepper_spheres_wheat/
mv data/cylinder_pepper_spheres_wheat/0* data/tmp_not_in_all/
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/crosseval_ME/train_cylinder_pepper_wheat_sep_gating_me.json \
  --dataset_dir 'data/cylinder_pepper_spheres_wheat/*_trackHistory_NothingDeleted.csv' \
  --result_path results/crosseval_ME/separation_prediction_gating_me_cylinder_pepper_wheat/ \
  --time_normalization_constant 15.71 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 3000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# spheres test
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network True \
  --config_path configs/crosseval_ME/train_cylinder_pepper_wheat_sep_gating_me.json \
  --dataset_dir 'data/tmp_not_in_all/*_trackHistory_NothingDeleted.csv' \
  --result_path results/crosseval_ME/separation_prediction_gating_me_test_sphere/ \
  --time_normalization_constant 15.71 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 3000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True \
  --evaluation_ratio 0.05 \
  --test_ratio 0.9

# cylinder_spheres_wheat
mv data/tmp_not_in_all/* data/cylinder_pepper_spheres_wheat/
mv data/cylinder_pepper_spheres_wheat/pepper* data/tmp_not_in_all/
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/crosseval_ME/train_cylinder_spheres_wheat_sep_gating_me.json \
  --dataset_dir 'data/cylinder_pepper_spheres_wheat/*_trackHistory_NothingDeleted.csv' \
  --result_path results/crosseval_ME/separation_prediction_gating_me_cylinder_spheres_wheat/ \
  --time_normalization_constant 15.71 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 3000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# pepper test
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network True \
  --config_path configs/crosseval_ME/train_cylinder_spheres_wheat_sep_gating_me.json \
  --dataset_dir 'data/tmp_not_in_all/*_trackHistory_NothingDeleted.csv' \
  --result_path results/crosseval_ME/separation_prediction_gating_me_test_pepper/ \
  --time_normalization_constant 15.71 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 3000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True \
  --evaluation_ratio 0.05 \
  --test_ratio 0.9

# pepper_spheres_wheat
mv data/tmp_not_in_all/* data/cylinder_pepper_spheres_wheat/
mv data/cylinder_pepper_spheres_wheat/cylinder* data/tmp_not_in_all/
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network False \
  --config_path configs/crosseval_ME/train_pepper_spheres_wheat_sep_gating_me.json \
  --dataset_dir 'data/cylinder_pepper_spheres_wheat/*_trackHistory_NothingDeleted.csv' \
  --result_path results/crosseval_ME/separation_prediction_gating_me_pepper_spheres_wheat/ \
  --time_normalization_constant 15.71 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 3000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True

# cylinder test
./main.py \
  --separation_prediction True \
  --tracking False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --is_loaded True \
  --is_loaded_gating_network True \
  --config_path configs/crosseval_ME/train_pepper_spheres_wheat_sep_gating_me.json \
  --dataset_dir 'data/tmp_not_in_all/*_trackHistory_NothingDeleted.csv' \
  --result_path results/crosseval_ME/separation_prediction_gating_me_test_cylinder/ \
  --time_normalization_constant 15.71 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 3000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 50 \
  --evaluate_mlp_mask False \
  --execute_evaluation True \
  --execute_multi_target_tracking False \
  --no_show True \
  --evaluation_ratio 0.05 \
  --test_ratio 0.9