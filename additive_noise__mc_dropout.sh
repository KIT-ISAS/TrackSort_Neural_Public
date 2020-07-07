python main.py \
     --is_loaded True \
     --model_path "models/mc4.h5" \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --additive_noise_stddev 0.0 \
     --mc_dropout True \
     --mc_samples 250 \
     --distance_confidence 0.999 \
     --calibrate False \
     --description "Noise=0. Calibrate=False. MC-Drop."

python main.py \
     --is_loaded True \
     --model_path "models/mc4.h5" \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --additive_noise_stddev 0.0 \
     --mc_dropout True \
     --mc_samples 250 \
     --distance_confidence 0.999 \
     --calibrate True \
     --description "Noise=0. Calibrate=True. MC-Drop."

python main.py \
     --is_loaded True \
     --model_path "models/mc4.h5" \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --additive_noise_stddev 0.0003 \
     --mc_dropout True \
     --mc_samples 250 \
     --distance_confidence 0.999 \
     --calibrate True \
     --description "Noise=0.0003 Calibrate=True. MC-Drop."


python main.py \
     --is_loaded True \
     --model_path "models/mc4.h5" \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --additive_noise_stddev 0.0005 \
     --mc_dropout True \
     --mc_samples 250 \
     --distance_confidence 0.999 \
     --calibrate True \
     --description "Noise=0.0005 Calibrate=True. MC-Drop."


python main.py \
     --is_loaded True \
     --model_path "models/mc4.h5" \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --additive_noise_stddev 0.0008 \
     --mc_dropout True \
     --mc_samples 250 \
     --distance_confidence 0.999 \
     --calibrate True \
     --description "Noise=0.0008 Calibrate=True. MC-Drop."


python main.py \
     --is_loaded True \
     --model_path "models/mc4.h5" \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --additive_noise_stddev 0.001 \
     --mc_dropout True \
     --mc_samples 250 \
     --distance_confidence 0.999 \
     --calibrate True \
     --description "Noise=0.001 Calibrate=True. MC-Drop."