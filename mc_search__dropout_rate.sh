# lstm 64-32-2
## dr=0.5
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 490 \
     --num_units_first_rnn 64 \
     --num_units_second_rnn 32 \
     --num_units_first_dense 0 \
     --num_units_second_dense 0 \
     --lr_decay_after_epochs 200 \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --dropout 0.5 \
     --augment_beginning True \
     --additive_noise_stddev 0.0001 \
     --mc_dropout True \
     --mc_samples 500 \
     --distance_threshold 20.0 \
     --run_association False \
     --description "mc_search__dropout_rate.sh: lstm 64-32-2, dr=0.5"


# lstm 64-32-2
## dr=0.2
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 490 \
     --num_units_first_rnn 64 \
     --num_units_second_rnn 32 \
     --num_units_first_dense 0 \
     --num_units_second_dense 0 \
     --lr_decay_after_epochs 200 \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --dropout 0.2 \
     --augment_beginning True \
     --additive_noise_stddev 0.0001 \
     --mc_dropout True \
     --mc_samples 500 \
     --distance_threshold 20.0 \
     --run_association False \
     --description "mc_search__dropout_rate.sh: lstm 64-32-2, dr=0.2"



# lstm 64-32-2
## dr=0.1
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 490 \
     --num_units_first_rnn 64 \
     --num_units_second_rnn 32 \
     --num_units_first_dense 0 \
     --num_units_second_dense 0 \
     --lr_decay_after_epochs 200 \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --dropout 0.1 \
     --augment_beginning True \
     --additive_noise_stddev 0.0001 \
     --mc_dropout True \
     --mc_samples 500 \
     --distance_threshold 20.0 \
     --run_association False \
     --description "mc_search__dropout_rate.sh: lstm 64-32-2, dr=0.1"


# lstm 64-32-2
## dr=0.01
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 490 \
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
     --description "mc_search__dropout_rate.sh: lstm 64-32-2, dr=0.01"

# lstm 64-32-2
## dr=0.005
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 490 \
     --num_units_first_rnn 64 \
     --num_units_second_rnn 32 \
     --num_units_first_dense 0 \
     --num_units_second_dense 0 \
     --lr_decay_after_epochs 200 \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --dropout 0.005 \
     --augment_beginning True \
     --additive_noise_stddev 0.0001 \
     --mc_dropout True \
     --mc_samples 500 \
     --distance_threshold 20.0 \
     --run_association False \
     --description "mc_search__dropout_rate.sh: lstm 64-32-2, dr=0.005"

# lstm 64-32-2
## dr=0.001
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 490 \
     --num_units_first_rnn 64 \
     --num_units_second_rnn 32 \
     --num_units_first_dense 0 \
     --num_units_second_dense 0 \
     --lr_decay_after_epochs 200 \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --dropout 0.001 \
     --augment_beginning True \
     --additive_noise_stddev 0.0001 \
     --mc_dropout True \
     --mc_samples 500 \
     --distance_threshold 20.0 \
     --run_association False \
     --description "mc_search__dropout_rate.sh: lstm 64-32-2, dr=0.001"


# lstm 64-32-2
## dr=0.0005
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 490 \
     --num_units_first_rnn 64 \
     --num_units_second_rnn 32 \
     --num_units_first_dense 0 \
     --num_units_second_dense 0 \
     --lr_decay_after_epochs 200 \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --dropout 0.0005 \
     --augment_beginning True \
     --additive_noise_stddev 0.0001 \
     --mc_dropout True \
     --mc_samples 500 \
     --distance_threshold 20.0 \
     --run_association False \
     --description "mc_search__dropout_rate.sh: lstm 64-32-2, dr=0.0005"