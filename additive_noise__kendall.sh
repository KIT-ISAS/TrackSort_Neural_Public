python main.py \
     --is_loaded True \
     --model_path "models/kendall2.h5" \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --additive_noise_stddev 0.0 \
     --kendall_loss True \
     --distance_confidence 0.999 \
     --calibrate False \
     --description "Noise=0. Calibrate=False. Kendall."

python main.py \
     --is_loaded True \
     --model_path "models/kendall2.h5" \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --additive_noise_stddev 0.0 \
     --kendall_loss True \
     --distance_confidence 0.999 \
     --calibrate True \
     --description "Noise=0. Calibrate=True. Kendall."

python main.py \
     --is_loaded True \
     --model_path "models/kendall2.h5" \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --additive_noise_stddev 0.0003 \
     --kendall_loss True \
     --distance_confidence 0.999 \
     --calibrate True \
     --description "Noise=0.0003 Calibrate=True. Kendall."


python main.py \
     --is_loaded True \
     --model_path "models/kendall2.h5" \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --additive_noise_stddev 0.0005 \
     --kendall_loss True \
     --distance_confidence 0.999 \
     --calibrate True \
     --description "Noise=0.0005 Calibrate=True. Kendall."


python main.py \
     --is_loaded True \
     --model_path "models/kendall2.h5" \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --additive_noise_stddev 0.0008 \
     --kendall_loss True \
     --distance_confidence 0.999 \
     --calibrate True \
     --description "Noise=0.0008 Calibrate=True. Kendall."


python main.py \
     --is_loaded True \
     --model_path "models/kendall2.h5" \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --additive_noise_stddev 0.001 \
     --kendall_loss True \
     --distance_confidence 0.999 \
     --calibrate True \
     --description "Noise=0.001 Calibrate=True. Kendall."