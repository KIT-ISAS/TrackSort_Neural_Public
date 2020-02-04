python main.py --model_path "models/DEM_model.h5" --dataset_dir "data/DEM_cylinder.csv" \
	--data_is_aligned False --is_loaded True  --rotate_columns True --run_hyperparameter_search False  \
	--normalization_constant 1.0 --test_noise_robustness True --distance_threshold 0.01