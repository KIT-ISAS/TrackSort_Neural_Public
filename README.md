# TrackSort Neural
Implementation of multitarget tracking and separation prediction for optical belt sorters.
**Key features:**
* Multiple experts for tracking and separation prediction
  * Kalman filters (KF) - constant velocity (CV) and constant acceleration (CA)
  * Multilayer Perceptrons (MLP)
  * Long-short term memory networks (LSTM)
* Expert combination methods to combine the predictions of multiple experts
  * Simple ensemble
  * Covariance weighting
  * SMAPE weighting
  * **Mixture of experts (ME)**
* Uncertainty prediction of the separation predictions
* Uncertainty evaluation and calibration with **new SENCE method**
* Large variety of evaluation functions for tracking and separation prediction

**Outline:**
* [Motivation](#motivation)
* [Code Structure](#code-structure)
* [Installation](#installation)
* [Common Run Commands](#common-run-commands)
* [Config files](#config-files)
* [Contribute](#contribute)
* [Credits](#credits)

## Motivation
This framework allows us to test and compare a vast variety of experts for the tasks of tracking and separation prediction in optical belt sorters.
The code includes every step from the data pipeline to the training and testing of the experts.
With this code we are able to train a state of the art ME approach that is able to outperform single experts in every sorting scenario and improves the sorting accuracy in dynamic scenarios significantly.
The implemented uncertainty prediction for the separation prediction not only improves the accuracy of the neural networks but is also a potential key input for a dynamic nozzle control.

## Code Structure
The idea of the expert combination in tracking can be explained with the following figure.
![ME structure](/images/ME_Structure_Tracking.png)
For every new measurement of a particle, the experts, here one KF, one LSTM, and one MLP, predict the next particle position.
The gating network then assigns weights to each expert prediction. The combined expert prediction is the weighted sum of expert predictions.
The combined prediction is used in the data association to match a prediction to a new measurement and the cycle restarts.

An example workflow with the most important classes and functions is displayed in the following figure.
![Sequence diagramm](/images/sequence_diagramm.svg)

## Installation

### Run with CPU

1. `. setup.sh` (creates a virtualenv and sources it -> `.` in the beginning is necessary)
2. `python main.py`

You can find the visualizations (step wise and as video) in the visualizations folder and you can set the hyperparams as described when typing `python main.py --help`

### Run with GPU (Docker)

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
     --dataset_dir "data/DEM_cylinders.csv" \
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
    

### Hyperparameter search with `python main.py`

ToDo: Explain --run_hyperparameter_search

## Notebooks:  Run with GPU (Docker)

Before implementing a functionality in the main code, we have tested and experimented in jupyter
notebooks. 

The `Notebook/`-folder contains short code samples to_

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

## Data store

The files are downloaded either from:
- private server  or
- ISAS i81server (you have to be in the institute's network)
