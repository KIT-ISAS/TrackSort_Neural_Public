
## dr=0.01
# lstm 32-32-2
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 510 \
     --num_units_first_rnn 32 \
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
     --description "mc_search__net_arch.sh: lstm 32-32-2, dr=0.01"



## dr=0.01
# lstm 16-16-2
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 510 \
     --num_units_first_rnn 16 \
     --num_units_second_rnn 16 \
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
     --description "mc_search__net_arch.sh: lstm 16-16-2, dr=0.01"


## dr=0.01
# lstm 8-8-2
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 510 \
     --num_units_first_rnn 8 \
     --num_units_second_rnn 8 \
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
     --description "mc_search__net_arch.sh: lstm 8-8-2, dr=0.01"


## dr=0.01
# lstm 8-4-2
python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 510 \
     --num_units_first_rnn 64 \
     --num_units_second_rnn 64 \
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
     --description "mc_search__net_arch.sh: lstm 8-4-2, dr=0.01"
