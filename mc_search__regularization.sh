# lstm 64-32-2
## reg=0.001
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 510 \
     --num_units_first_rnn 64 \
     --num_units_second_rnn 32 \
     --num_units_first_dense 0 \
     --num_units_second_dense 0 \
     --lr_decay_after_epochs 200 \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --dropout 0.01 \
     --augment_beginning True \
     --additive_noise_stddev 0.0001 \
     --mc_dropout True \
     --mc_samples 500 \
     --distance_threshold 20.0 \
     --run_association False \
     --regularization 0.001 \
     --description "mc_search__regularization.sh: lstm 64-32-2, reg=0.001"

# lstm 64-32-2
## reg=0.0001
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 510 \
     --num_units_first_rnn 64 \
     --num_units_second_rnn 32 \
     --num_units_first_dense 0 \
     --num_units_second_dense 0 \
     --lr_decay_after_epochs 200 \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --dropout 0.01 \
     --augment_beginning True \
     --additive_noise_stddev 0.0001 \
     --mc_dropout True \
     --mc_samples 500 \
     --distance_threshold 20.0 \
     --run_association False \
     --regularization 0.0001 \
     --description "mc_search__regularization.sh: lstm 64-32-2, reg=0.0001"


# lstm 64-32-2
## reg=0.00001
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 510 \
     --num_units_first_rnn 64 \
     --num_units_second_rnn 32 \
     --num_units_first_dense 0 \
     --num_units_second_dense 0 \
     --lr_decay_after_epochs 200 \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --dropout 0.01 \
     --augment_beginning True \
     --additive_noise_stddev 0.0001 \
     --mc_dropout True \
     --mc_samples 500 \
     --distance_threshold 20.0 \
     --run_association False \
     --regularization 0.00001 \
     --description "mc_search__regularization.sh: lstm 64-32-2, reg=0.00001"



# lstm 64-32-2
## reg=0.000001
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 510 \
     --num_units_first_rnn 64 \
     --num_units_second_rnn 32 \
     --num_units_first_dense 0 \
     --num_units_second_dense 0 \
     --lr_decay_after_epochs 200 \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --dropout 0.01 \
     --augment_beginning True \
     --additive_noise_stddev 0.0001 \
     --mc_dropout True \
     --mc_samples 500 \
     --distance_threshold 20.0 \
     --run_association False \
     --regularization 0.000001 \
     --description "mc_search__regularization.sh: lstm 64-32-2, reg=0.000001"

