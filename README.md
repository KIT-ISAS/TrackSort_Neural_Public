# NextStep-RNN

**Content:**

- Using a RNN to predict next measurement of tracks
- Using a RNN to make separation predictions
- Python implementation of data assocation (local and global nearest neighbour)

## Run with CPU

1. `. setup.sh` (creates a virtualenv and sources it -> `.` in the beginning is necessary)
2. `python main.py`

You can find the visualizations (step wise and as video) in the visualizations folder and you can set the hyperparams as described when typing `python main.py --help`

## Run with GPU (Docker)

1. Pull docker image `docker pull tensorflow/tensorflow:2.1.0-gpu-py3`
2. Clone repo: `git clone https://github.com/sidney1505/next_step_rnn`
3. Run docker with mounted dir: `docker run -it -v $PWD:/tf -w /tf --gpus "device=0" tensorflow/tensorflow:2.1.0-gpu-py3`
4. Inside the container: `cd next_step_rnn`
5. Inside: `. setup.sh`
6. Inside: `python main.py ...`

## `python main.py`

This is the main starting point and it has a lot of settings.
See: `python main.py --help`

### Example configurations for ` python main.py`

- Evaluate a pretrained model on the DEM dataset

  ```shell script
  python main.py \
    --is_loaded True \
    --model_path "models/DEM_model.h5" \
    --dataset_dir "data/DEM_cylinder.csv" \
    --data_is_aligned False \
    --rotate_columns True \
    --normalization_constant 1.0
  ```

- Evaluate a pretrained model on the Pfeffer data

  ```shell script
  python main.py \
     --is_loaded True \
     --model_path "models/rnn_model_fake_data.h5" \
     --dataset_dir "data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv" \
     --data_is_aligned True \
     --normalization_constant 2000.0 
  ```

- Train a lstm-16-16 on the Pfeffer data
  ```shell script
  python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 10 \
     --num_units_first_rnn 16 \
     --num_units_second_rnn 16 \
     --lr_decay_after_epochs 150 \
     --dataset_dir "data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv" \
     --data_is_aligned True \
     --normalization_constant 2000.0 
  ```
  
- Train a lstm-16-16 on the DEM data
  ```shell script
  python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 10 \
     --num_units_first_rnn 16 \
     --num_units_second_rnn 16 \
     --lr_decay_after_epochs 150 \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True
  ```
  
- Evaluate the separation prediction on the DEM cuboids

  ```shell script
  python main.py \
     --is_loaded False \
     --dataset_dir "data/DEM_cuboids.csv" \
     --data_is_aligned False \
     --rotate_columns True \
     --normalization_constant 1.0 \
     --separation_prediction True \
     --virtual_nozzle_array_x_position 0.7 \
     --virtual_belt_edge_x_position 0.55
  ```
  
- Evaluate impact of noise

  ```shell script
  python main.py \
    --is_loaded True \
    --model_path "models/DEM_model.h5" \
    --dataset_dir "data/DEM_cylinder.csv" \
    --data_is_aligned False \
    --rotate_columns True \
    --normalization_constant 1.0 \
    --test_noise_robustness True
  ```
  
- Train MC Dropout

    ```shell script
  
  python main.py \
     --is_loaded False \
     --num_train_epochs 500 \
     --evaluate_every_n_epochs 10 \
     --num_units_first_rnn 128 \
     --num_units_second_rnn 0 \
     --lr_decay_after_epochs 150 \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --dropout 0.5 \
     --augment_beginning True \
     --additive_noise_stddev 0.0001 \
     --mc_dropout True \
     --mc_samples 5 \
     --distance_threshold 20.0

    ```
  
- Train with negative log likelihood

    ```shell script
    python main.py \
     --is_loaded False \
     --num_train_epochs 1000 \
     --evaluate_every_n_epochs 10 \
     --num_units_first_rnn 64 \
     --num_units_second_rnn 32 \
     --lr_decay_after_epochs 150 \
     --dataset_dir "data/DEM_cylinder.csv" \
     --data_is_aligned False \
     --normalization_constant 1.0 \
     --rotate_columns True \
     --dropout 0.2 \
     --augment_beginning True \
     --additive_noise_stddev 0.0001 \
     --kendall_loss True \
     --distance_threshold 20.0
    ```
  
    

### Hyperparameter search with `python main.py`

ToDo: Explain --run_hyperparameter_search

## Notebooks:  Run with GPU (Docker)

Before implementing a functionality in the main code, we have tested and experimented in jupyter
notebooks. 

The `Notebook/`-folder contains short code samples to

- train a NextStep-RNN
- Use the ModelManager
- Load FakeData
- Load CsvData
- Experiments with curriculum learning
- Experiments with Bayesian learning
- The broader line of the data association
- Gridsearch for models
- Evaluation of separation models

Follow these instructions to run notebooks on the gpu pc at the institute.

1. `docker pull tensorflow/tensorflow:2.1.0-gpu-py3-jupyter`
2. Clone repo: `git clone https://github.com/sidney1505/next_step_rnn`
3. `docker run -it -p 8888:8888 -v $PWD:/tf --gpus "device=0" tensorflow/tensorflow:2.1.0-gpu-py3-jupyter`
4. (ssh tunneling on remote machine: `ssh -L 8888:127.0.0.1:8888 proprak7@i81-gpu-server`)

Make the venv available as a kernel:

1. source the venv
2. `python3 -m pip install ipykernel` 
3. `python3 -m ipykernel install --name "local-venv" --user`

## Data store

The files are downloaded either from:
- private server  or
- ISAS i81server (you have to be in the institute's network)