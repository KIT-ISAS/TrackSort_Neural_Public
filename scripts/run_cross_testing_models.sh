#!/bin/sh

cd ..

### Zylinder
: '
python3 main.py \
        --is_loaded True \
        --config_path configs/test_cylinder_all.json \
        --dataset_dir 'data/Zylinder/trackSortResultZylinder/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/cylinder_all/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False

python3 main.py \
        --is_loaded True \
        --config_path configs/test_cylinder_all.json \
        --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/cylinder_models_on_pepper_data/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False

python3 main.py \
        --is_loaded True \
        --config_path configs/test_cylinder_all.json \
        --dataset_dir 'data/Kugeln/trackSortResultKugeln/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/cylinder_models_on_spheres_data/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False

python3 main.py \
        --is_loaded True \
        --config_path configs/test_cylinder_all.json \
        --dataset_dir 'data/Weizen/trackSortResultWeizen/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/cylinder_models_on_wheat_data/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False

### Pepper
python3 main.py \
        --is_loaded True \
        --config_path configs/test_pepper_all.json \
        --dataset_dir 'data/Zylinder/trackSortResultZylinder/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/pepper_models_on_cylinder_data/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False

python3 main.py \
        --is_loaded True \
        --config_path configs/test_pepper_all.json \
        --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/pepper_all/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False

python3 main.py \
        --is_loaded True \
        --config_path configs/test_pepper_all.json \
        --dataset_dir 'data/Kugeln/trackSortResultKugeln/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/pepper_models_on_spheres_data/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False
'
python3 main.py \
        --is_loaded True \
        --config_path configs/test_pepper_all.json \
        --dataset_dir 'data/Weizen/trackSortResultWeizen/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/pepper_models_on_wheat_data/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False
exit()

### Spheres
python3 main.py \
        --is_loaded True \
        --config_path configs/test_spheres_all.json \
        --dataset_dir 'data/Zylinder/trackSortResultZylinder/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/spheres_models_on_cylinder_data/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False
exit()

python3 main.py \
        --is_loaded True \
        --config_path configs/test_spheres_all.json \
        --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/spheres_models_on_pepper_data/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False
exit()

python3 main.py \
        --is_loaded True \
        --config_path configs/test_spheres_all.json \
        --dataset_dir 'data/Kugeln/trackSortResultKugeln/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/spheres_all/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False
exit()

python3 main.py \
        --is_loaded True \
        --config_path configs/test_spheres_all.json \
        --dataset_dir 'data/Weizen/trackSortResultWeizen/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/spheres_models_on_wheat_data/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False
exit()

### Wheat
python3 main.py \
        --is_loaded True \
        --config_path configs/test_wheat_all.json \
        --dataset_dir 'data/Zylinder/trackSortResultZylinder/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/wheat_models_on_cylinder_data/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False
exit()

python3 main.py \
        --is_loaded True \
        --config_path configs/test_wheat_all.json \
        --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/wheat_models_on_pepper_data/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False
exit()

python3 main.py \
        --is_loaded True \
        --config_path configs/test_wheat_all.json \
        --dataset_dir 'data/Kugeln/trackSortResultKugeln/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/wheat_models_on_spheres_data/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False
exit()

python3 main.py \
        --is_loaded True \
        --config_path configs/test_wheat_all.json \
        --dataset_dir 'data/Weizen/trackSortResultWeizen/*_trackHistory_NothingDeleted.csv' \
        --result_path 'results/wheat_all/' \
        --execute_evaluation True \
        --execute_multi_target_tracking False
exit()
